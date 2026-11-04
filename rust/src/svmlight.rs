//! Parallel parser for the LETOR / SVMlight ranking format.
//!
//! Each line is `<label> qid:<id> <index>:<value> ... [# comment]` with 1-based feature indices.
//! The text is split into chunks at line boundaries, chunks are parsed in parallel, and the rows
//! are written into one dense row-major `f32` matrix (absent features are 0, as in the format).

use rayon::prelude::*;

/// A parsed ranking file: dense row-major features plus labels and query ids per row.
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub labels: Vec<f32>,
    pub qids: Vec<i64>,
    pub features: Vec<f32>,
    pub n_features: usize,
}

struct ChunkRows {
    labels: Vec<f32>,
    qids: Vec<i64>,
    entries: Vec<(u32, f32)>,
    row_ends: Vec<usize>,
    max_index: u32,
}

fn parse_line(line: &str, line_no: usize, out: &mut ChunkRows) -> Result<(), String> {
    let content = match line.find('#') {
        Some(pos) => &line[..pos],
        None => line,
    };
    let mut tokens = content.split_ascii_whitespace();
    let Some(label_token) = tokens.next() else {
        return Ok(());
    };
    let label: f32 = label_token
        .parse()
        .map_err(|_| format!("line {line_no}: invalid label {label_token:?}"))?;
    let qid_token = tokens
        .next()
        .ok_or_else(|| format!("line {line_no}: missing qid"))?;
    let qid: i64 = qid_token
        .strip_prefix("qid:")
        .ok_or_else(|| format!("line {line_no}: expected qid:<id>, got {qid_token:?}"))?
        .parse()
        .map_err(|_| format!("line {line_no}: invalid qid {qid_token:?}"))?;
    for token in tokens {
        let (index, value) = token
            .split_once(':')
            .ok_or_else(|| format!("line {line_no}: invalid feature {token:?}"))?;
        let index: u32 = index
            .parse()
            .map_err(|_| format!("line {line_no}: invalid feature index {token:?}"))?;
        if index == 0 {
            return Err(format!("line {line_no}: feature indices start at 1"));
        }
        let value: f32 = value
            .parse()
            .map_err(|_| format!("line {line_no}: invalid feature value {token:?}"))?;
        out.max_index = out.max_index.max(index);
        out.entries.push((index - 1, value));
    }
    out.labels.push(label);
    out.qids.push(qid);
    out.row_ends.push(out.entries.len());
    Ok(())
}

fn chunk_bounds(text: &[u8], target_chunks: usize) -> Vec<(usize, usize)> {
    let n = text.len();
    let step = (n / target_chunks.max(1)).max(1 << 16);
    let mut bounds = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = (start + step).min(n);
        while end < n && text[end - 1] != b'\n' {
            end += 1;
        }
        bounds.push((start, end));
        start = end;
    }
    bounds
}

/// Parse LETOR text. `n_features = 0` infers the width from the largest feature index.
pub fn parse(text: &str, n_features: usize) -> Result<Parsed, String> {
    let bytes = text.as_bytes();
    let bounds = chunk_bounds(bytes, rayon::current_num_threads() * 4);
    // Line numbers are reported per chunk start, so count newlines before each chunk.
    let mut first_line = Vec::with_capacity(bounds.len());
    let mut lines_before = 0usize;
    for &(s, e) in &bounds {
        first_line.push(lines_before + 1);
        lines_before += bytes[s..e].iter().filter(|&&b| b == b'\n').count();
    }
    let chunks: Vec<ChunkRows> = bounds
        .par_iter()
        .zip(first_line.par_iter())
        .map(|(&(s, e), &line0)| {
            let mut rows = ChunkRows {
                labels: Vec::new(),
                qids: Vec::new(),
                entries: Vec::new(),
                row_ends: Vec::new(),
                max_index: 0,
            };
            for (offset, line) in text[s..e].lines().enumerate() {
                parse_line(line, line0 + offset, &mut rows)?;
            }
            Ok(rows)
        })
        .collect::<Result<_, String>>()?;
    let max_index = chunks.iter().map(|c| c.max_index).max().unwrap_or(0) as usize;
    let d = if n_features == 0 {
        max_index
    } else {
        if max_index > n_features {
            return Err(format!(
                "feature index {max_index} exceeds n_features={n_features}"
            ));
        }
        n_features
    };
    let n: usize = chunks.iter().map(|c| c.labels.len()).sum();
    let mut features = vec![0f32; n * d];
    let mut slices: Vec<&mut [f32]> = Vec::with_capacity(chunks.len());
    let mut rest: &mut [f32] = &mut features;
    for c in &chunks {
        let (head, tail) = rest.split_at_mut(c.labels.len() * d);
        slices.push(head);
        rest = tail;
    }
    slices
        .into_par_iter()
        .zip(chunks.par_iter())
        .for_each(|(dense, c)| {
            let mut begin = 0;
            for (row, &end) in c.row_ends.iter().enumerate() {
                let base = row * d;
                for &(j, v) in &c.entries[begin..end] {
                    dense[base + j as usize] = v;
                }
                begin = end;
            }
        });
    let mut labels = Vec::with_capacity(n);
    let mut qids = Vec::with_capacity(n);
    for c in chunks {
        labels.extend(c.labels);
        qids.extend(c.qids);
    }
    Ok(Parsed {
        labels,
        qids,
        features,
        n_features: d,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dense_rows_and_comments() {
        let text = "2 qid:10 1:0.5 3:2 # doc a\n0 qid:10 2:1.5\n\n1 qid:11 1:-1e-3\n";
        let p = parse(text, 0).unwrap();
        assert_eq!(p.n_features, 3);
        assert_eq!(p.labels, vec![2.0, 0.0, 1.0]);
        assert_eq!(p.qids, vec![10, 10, 11]);
        assert_eq!(
            p.features,
            vec![0.5, 0.0, 2.0, 0.0, 1.5, 0.0, -1e-3, 0.0, 0.0]
        );
    }

    #[test]
    fn explicit_width_pads_and_validates() {
        let p = parse("1 qid:1 2:3\n", 4).unwrap();
        assert_eq!(p.features, vec![0.0, 3.0, 0.0, 0.0]);
        assert!(parse("1 qid:1 5:3\n", 4).is_err());
    }

    #[test]
    fn reports_bad_lines() {
        assert!(parse("x qid:1 1:2\n", 0).unwrap_err().contains("line 1"));
        assert!(parse("1 1:2\n", 0).unwrap_err().contains("qid"));
        assert!(parse("1 qid:1 0:2\n", 0).is_err());
        assert!(parse("1 qid:1 1:z\n", 0).is_err());
        assert!(parse("1 qid:1 12\n", 0).is_err());
        assert!(parse("1 qid:a 1:1\n", 0).is_err());
        assert!(parse("1 qid:1 a:1\n", 0).is_err());
    }

    #[test]
    fn many_chunks_keep_row_order() {
        let mut text = String::new();
        for i in 0..50_000 {
            text.push_str(&format!("{} qid:{} 1:{} 2:{}\n", i % 5, i / 7, i, -i));
        }
        let p = parse(&text, 0).unwrap();
        assert_eq!(p.labels.len(), 50_000);
        for i in [0usize, 1, 12_345, 49_999] {
            assert_eq!(p.features[2 * i], i as f32);
            assert_eq!(p.qids[i], (i / 7) as i64);
        }
    }
}

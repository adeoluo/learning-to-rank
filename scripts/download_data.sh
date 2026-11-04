#!/usr/bin/env bash
# Download the public learning-to-rank datasets used by the benchmarks and convert them.
#
#   MSLR-WEB10K: the Hugging Face parquet mirror of Fold 1 (philipphager/MSLR-WEB10k, 431 MB),
#                converted to numpy arrays with all five partitions (see datasets.py).
#   Istella-S:   the Istella LETOR archive (472 MB), streamed into 255-bin uint8 features.
#
# Data goes to $LTR_DATA (default ../.bench-data/learning-to-rank), never into the repository.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${LTR_DATA:-$ROOT/../.bench-data/learning-to-rank}"
mkdir -p "$DATA"
HF="https://huggingface.co/datasets/philipphager/MSLR-WEB10k/resolve/main"

for split in train validation test; do
  if [[ ! -s "$DATA/hf_$split.parquet" ]]; then
    echo "downloading MSLR-WEB10K $split"
    curl -fL --retry 3 -o "$DATA/hf_$split.parquet" "$HF/$split.parquet"
  fi
done
if [[ ! -s "$DATA/mslr10k/X.npy" ]]; then
  uv run --extra data python -c "from learning_to_rank.datasets import prepare_mslr; print(prepare_mslr('$DATA', '$DATA/mslr10k'))"
fi

if [[ "${SKIP_ISTELLA:-0}" != "1" ]]; then
  if [[ ! -s "$DATA/istella-s-letor.tar.gz" ]]; then
    echo "downloading Istella-S (research use under the Istella LETOR licence)"
    curl -fL --retry 3 -o "$DATA/istella-s-letor.tar.gz" http://library.istella.it/dataset/istella-s-letor.tar.gz
  fi
fi
echo "data ready in $DATA"

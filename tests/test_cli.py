from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from learning_to_rank.cli import load_model, main
from learning_to_rank.data import synthetic, to_svmlight
from learning_to_rank.datasets import MSLR_FEATURES


@pytest.fixture(scope="module")
def files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("letor")
    out = {}
    for name, seed in (("train", 1), ("valid", 2), ("test", 3)):
        p = d / f"{name}.txt"
        p.write_text(to_svmlight(synthetic(80, (5, 20), 6, seed=seed)))
        out[name] = p
    return out


@pytest.fixture(scope="module")
def mslr_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny directory in the prepare-mslr layout (five partitions)."""

    d = tmp_path_factory.mktemp("mslr")
    rng = np.random.default_rng(0)
    qids = np.array([q for q in range(1, 15 * 40) if q % 15 in (1, 4, 7, 10, 13)])
    sizes = rng.integers(8, 20, size=qids.shape[0])
    n = int(sizes.sum())
    X = rng.normal(size=(n, MSLR_FEATURES)).astype(np.float32)
    y = np.clip(np.round(X[:, 0] + rng.normal(scale=0.5, size=n)), 0, 4).astype(np.int8)
    qid = np.repeat(qids, sizes).astype(np.int32)
    part = np.array([{1: 1, 4: 2, 7: 3, 10: 4, 13: 5}[q % 15] for q in qid], dtype=np.int8)
    order = np.lexsort((qid, part))
    np.save(d / "X.npy", X[order])
    np.save(d / "y.npy", y[order])
    np.save(d / "qid.npy", qid[order])
    np.save(d / "part.npy", part[order])
    return d


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    assert "learning-to-rank" in capsys.readouterr().out


def test_train_evaluate_compare_files(
    tmp_path: Path, files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert (
        main(
            [
                "train",
                "--train-file",
                str(files["train"]),
                "--valid-file",
                str(files["valid"]),
                "--model",
                str(a),
                "--trees",
                "20",
                "--log-every",
                "10",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--train-file",
                str(files["train"]),
                "--model",
                str(b),
                "--trees",
                "5",
                "--leaves",
                "3",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "tree    20  valid ndcg@10" in out and "saved" in out
    assert (
        main(["evaluate", "--file", str(files["test"]), "--model", str(a), str(b), "--json"]) == 0
    )
    out = capsys.readouterr().out
    assert "ndcg@10" in out and '"map"' in out
    assert (
        main(["compare", "--file", str(files["test"]), str(a), str(b), "--permutations", "200"])
        == 0
    )
    out = capsys.readouterr().out
    assert "randomization test p" in out and "A wins" in out
    _model, n_features, info = load_model(a)
    assert n_features == 6 and info["type"] == "lambdamart" and info["n_trees"] == 20


def test_mslr_commands(tmp_path: Path, mslr_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    m = tmp_path / "m.json"
    assert (
        main(
            [
                "train",
                "--data",
                str(mslr_dir),
                "--fold",
                "2",
                "--model",
                str(m),
                "--trees",
                "10",
                "--early-stopping",
                "5",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "train: 120 queries" in out and "err@10" in out
    assert main(["evaluate", "--data", str(mslr_dir), "--split", "valid", "--model", str(m)]) == 0
    assert "40 queries" in capsys.readouterr().out
    assert (
        main(
            ["clicks", "--data", str(mslr_dir), "--sessions", "5000", "--production-queries", "10"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "swap estimate" in out and "true theta" in out and "pbm" in out
    assert (
        main(
            [
                "clicks",
                "--data",
                str(mslr_dir),
                "--sessions",
                "2000",
                "--click-model",
                "cascade",
                "--swap-fraction",
                "0",
            ]
        )
        == 0
    )


def test_neural_train_and_load(
    tmp_path: Path, files: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("torch")
    p = tmp_path / "nn.pt"
    assert (
        main(
            [
                "train",
                "--train-file",
                str(files["train"]),
                "--valid-file",
                str(files["valid"]),
                "--ranker",
                "neural",
                "--epochs",
                "2",
                "--model",
                str(p),
            ]
        )
        == 0
    )
    assert "epoch   2" in capsys.readouterr().out
    _, n, info = load_model(p)
    assert n == 6 and info["type"] == "neural"


def test_prepare_requires_args() -> None:
    with pytest.raises(SystemExit):
        main(["prepare-mslr"])
    with pytest.raises(SystemExit):
        main(["evaluate", "--model", "x.json"])

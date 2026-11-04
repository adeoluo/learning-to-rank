from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from learning_to_rank import LambdaMART, RankingData
from learning_to_rank.service import MAX_DOCUMENTS, App, make_handler


@pytest.fixture(scope="module")
def app(medium: tuple[RankingData, ...]) -> App:
    train, _, _ = medium
    model = LambdaMART(n_estimators=10).fit(train)
    return App(model, train.n_features, {"type": "lambdamart"})


def call(app: App, method: str, path: str, payload: object | None = None) -> tuple[int, dict]:  # type: ignore[type-arg]
    body = json.dumps(payload).encode() if payload is not None else b""
    status, ctype, out = app.handle(method, path, body)
    return status, (json.loads(out) if ctype == "application/json" else {"text": out})


def test_health_ready_model(app: App) -> None:
    assert call(app, "GET", "/health")[1]["status"] == "ok"
    assert call(app, "GET", "/ready") == (200, {"ready": True})
    status, info = call(app, "GET", "/model")
    assert status == 200 and info["type"] == "lambdamart" and info["n_features"] == 20
    empty = App(None, 3)
    assert call(empty, "GET", "/ready")[0] == 503
    assert call(empty, "POST", "/rank", {"documents": [[0, 0, 0]]})[0] == 503


def test_rank_orders_by_score(app: App, medium: tuple[RankingData, ...]) -> None:
    _, _, test = medium
    rows = test.X[:12].tolist()
    status, out = call(
        app, "POST", "/rank", {"documents": rows, "ids": [f"d{i}" for i in range(12)], "top_k": 5}
    )
    assert status == 200
    res = out["results"]
    assert len(res) == 5 and [r["rank"] for r in res] == [1, 2, 3, 4, 5]
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert res[0]["id"] == f"d{res[0]['index']}"
    _, raw = call(app, "POST", "/score", {"documents": rows})
    assert np.argmax(raw["scores"]) == res[0]["index"]


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"documents": []}, 400),
        ({"documents": [[1, 2]]}, 400),
        ({"documents": [["a"] * 20]}, 400),
        ({"documents": [[float("nan")] * 20]}, 400),
        ({"documents": [[0] * 20], "ids": ["a", "b"]}, 400),
        ({"documents": [[0] * 20], "top_k": 0}, 400),
    ],
)
def test_validation(app: App, payload: dict, status: int) -> None:  # type: ignore[type-arg]
    assert call(app, "POST", "/rank", payload)[0] == status


def test_errors_and_metrics(app: App) -> None:
    assert app.handle("POST", "/rank", b"{bad")[0] == 400
    assert app.handle("POST", "/rank", b"[1]")[0] == 400
    assert call(app, "GET", "/nope")[0] == 404
    assert call(app, "POST", "/nope", {})[0] == 404
    assert call(app, "DELETE", "/rank")[0] == 405
    too_many = {"documents": [[0]] * (MAX_DOCUMENTS + 1)}
    assert call(app, "POST", "/score", too_many)[0] == 413
    text = call(app, "GET", "/metrics")[1]["text"]
    assert 'ltr_requests_total{route="/rank",status="200"}' in text
    assert "ltr_documents_scored_total" in text and "ltr_model_loaded 1" in text


def test_over_http(app: App) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, max_body=10_000))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as r:
            assert json.loads(r.read())["status"] == "ok"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/score",
            data=json.dumps({"documents": [[0.0] * 20]}).encode(),
            method="POST",
        )
        with urllib.request.urlopen(req) as r:
            assert len(json.loads(r.read())["scores"]) == 1
        big = urllib.request.Request(
            f"http://127.0.0.1:{port}/score", data=b"x" * 20_000, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(big)
        assert err.value.code == 413
    finally:
        server.shutdown()
        server.server_close()

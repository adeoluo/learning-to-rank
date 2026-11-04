"""HTTP ranking service (standard library).

Routes:

* ``GET /health``: liveness; ``GET /ready``: whether a model is loaded;
  ``GET /metrics``: Prometheus text; ``GET /model``: model summary and top features.
* ``POST /rank`` ``{"documents": [[f1, ..., fd], ...], "ids": [...], "top_k": 10}``: scores the
  documents of one query and returns them in ranked order (ties keep input order).
* ``POST /score`` ``{"documents": [[...], ...]}``: raw scores, in input order.

Inputs are validated (shape, numeric values, at most ``MAX_DOCUMENTS`` rows per request) and
errors come back as JSON with a 4xx status.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from learning_to_rank import __version__
from learning_to_rank.observability import Metrics

MAX_DOCUMENTS = 100_000


class Scorer(Protocol):
    def predict(self, X: NDArray[np.floating]) -> NDArray[np.float64]: ...


class HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _matrix(payload: dict[str, Any], n_features: int) -> NDArray[np.float32]:
    docs = payload.get("documents")
    if not isinstance(docs, list) or not docs:
        raise HTTPError(400, "'documents' must be a non-empty list of feature lists")
    if len(docs) > MAX_DOCUMENTS:
        raise HTTPError(413, f"at most {MAX_DOCUMENTS} documents per request")
    if not all(isinstance(row, list) and len(row) == n_features for row in docs):
        raise HTTPError(400, f"every document must be a list of {n_features} numbers")
    try:
        X = np.asarray(docs, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise HTTPError(400, "feature values must be numbers") from exc
    if not np.isfinite(X).all():
        raise HTTPError(400, "feature values must be finite")
    return X


class App:
    """Request handling, independent of the socket layer (easy to test)."""

    ROUTES = ("/health", "/ready", "/metrics", "/model", "/rank", "/score")

    def __init__(
        self,
        model: Scorer | None,
        n_features: int,
        info: dict[str, Any] | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self.model = model
        self.n_features = n_features
        self.info = info or {}
        self.metrics = metrics or Metrics()

    def handle(self, method: str, path: str, body: bytes) -> tuple[int, str, str]:
        """Return (status, content type, body)."""

        start = time.perf_counter()
        route = path.split("?", 1)[0]
        try:
            status, ctype, out = self._dispatch(method, route, body)
        except HTTPError as exc:
            status, ctype, out = exc.status, "application/json", json.dumps({"error": str(exc)})
        known = route if route in self.ROUTES else "other"
        self.metrics.observe_request(known, status, time.perf_counter() - start)
        return status, ctype, out

    def _json(self, body: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise HTTPError(400, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise HTTPError(400, "the request body must be a JSON object")
        return payload

    def _dispatch(self, method: str, route: str, body: bytes) -> tuple[int, str, str]:
        ok = "application/json"
        if method == "GET":
            if route == "/health":
                return 200, ok, json.dumps({"status": "ok", "version": __version__})
            if route == "/ready":
                ready = self.model is not None
                return (200 if ready else 503), ok, json.dumps({"ready": ready})
            if route == "/metrics":
                self.metrics.set_gauge("model_loaded", float(self.model is not None))
                return 200, "text/plain; version=0.0.4", self.metrics.render()
            if route == "/model":
                return 200, ok, json.dumps({"n_features": self.n_features, **self.info})
            raise HTTPError(404, "not found")
        if method != "POST":
            raise HTTPError(405, "method not allowed")
        if route not in ("/rank", "/score"):
            raise HTTPError(404, "not found")
        if self.model is None:
            raise HTTPError(503, "no model is loaded")
        payload = self._json(body)
        X = _matrix(payload, self.n_features)
        t0 = time.perf_counter()
        scores = self.model.predict(X)
        self.metrics.inc("documents_scored", X.shape[0])
        self.metrics.inc("scoring_seconds", time.perf_counter() - t0)
        if route == "/score":
            return 200, ok, json.dumps({"scores": scores.tolist()})
        ids = payload.get("ids")
        if ids is not None and (not isinstance(ids, list) or len(ids) != X.shape[0]):
            raise HTTPError(400, "'ids' must be a list with one id per document")
        top_k = payload.get("top_k", X.shape[0])
        if not isinstance(top_k, int) or top_k < 1:
            raise HTTPError(400, "'top_k' must be a positive integer")
        order = np.argsort(-scores, kind="stable")[:top_k]
        ranked = [
            {
                "rank": r + 1,
                "index": int(i),
                "score": float(scores[i]),
                **({"id": ids[i]} if ids else {}),
            }
            for r, i in enumerate(order)
        ]
        self.metrics.inc("queries_ranked")
        return 200, ok, json.dumps({"results": ranked})


def make_handler(app: App, max_body: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = f"learning-to-rank/{__version__}"

        def _send(self, status: int, ctype: str, body: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._send(*app.handle("GET", self.path, b""))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_body:
                self._send(413, "application/json", json.dumps({"error": "body too large"}))
                return
            self._send(*app.handle("POST", self.path, self.rfile.read(length)))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def serve(
    app: App, host: str, port: int, max_body: int = 64 * 1024 * 1024
) -> None:  # pragma: no cover - blocking
    server = ThreadingHTTPServer((host, port), make_handler(app, max_body))
    print(f"serving on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

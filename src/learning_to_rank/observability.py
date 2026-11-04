"""Thread-safe service metrics in the Prometheus text exposition format."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    """Request counts by route and status, latency sums, counters and gauges."""

    def __init__(self, prefix: str = "ltr") -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, int], int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}

    def observe_request(self, route: str, status: int, seconds: float) -> None:
        with self._lock:
            self._requests[(route, status)] += 1
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] += value

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def render(self) -> str:
        p = self.prefix
        with self._lock:
            lines = [
                f"# HELP {p}_requests_total HTTP requests by route and status.",
                f"# TYPE {p}_requests_total counter",
            ]
            for (route, status), n in sorted(self._requests.items()):
                lines.append(f'{p}_requests_total{{route="{route}",status="{status}"}} {n}')
            lines += [
                f"# HELP {p}_request_seconds HTTP request latency.",
                f"# TYPE {p}_request_seconds summary",
            ]
            for route in sorted(self._latency_count):
                lines.append(
                    f'{p}_request_seconds_sum{{route="{route}"}} {self._latency_sum[route]:.6f}'
                )
                lines.append(
                    f'{p}_request_seconds_count{{route="{route}"}} {self._latency_count[route]}'
                )
            for name, value in sorted(self._counters.items()):
                lines += [f"# TYPE {p}_{name}_total counter", f"{p}_{name}_total {value:g}"]
            for name, value in sorted(self._gauges.items()):
                lines += [f"# TYPE {p}_{name} gauge", f"{p}_{name} {value:g}"]
        return "\n".join(lines) + "\n"

"""Bounded local readiness load probe; not a clinical workflow benchmark."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import math
import sys
import time
from collections import Counter
from statistics import median
from typing import TypedDict
from urllib.parse import urlsplit

import httpx

MAX_REQUESTS = 10_000
MAX_CONCURRENCY = 100
ALLOWED_PATHS = {"/health", "/ready"}


class LoadProbeReport(TypedDict):
    target: str
    requests: int
    concurrency: int
    elapsedSeconds: float
    requestsPerSecond: float
    statusCounts: dict[str, int]
    failures: int
    latencyMs: dict[str, float]
    interpretation: str


def validate_target_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ALLOWED_PATHS
    ):
        raise ValueError("target must be a local /health or /ready URL without credentials")
    hostname = parsed.hostname.lower()
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname == "localhost"
    if not is_loopback:
        raise ValueError("load probes are restricted to loopback targets")
    return value


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("at least one latency sample is required")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(values)
    position = math.ceil(percentile_value / 100 * len(ordered)) - 1
    return ordered[max(0, position)]


async def run_probe(target: str, requests: int, concurrency: int) -> LoadProbeReport:
    validate_target_url(target)
    if not 1 <= requests <= MAX_REQUESTS:
        raise ValueError(f"requests must be between 1 and {MAX_REQUESTS}")
    if not 1 <= concurrency <= min(MAX_CONCURRENCY, requests):
        raise ValueError(f"concurrency must be between 1 and {min(MAX_CONCURRENCY, requests)}")

    semaphore = asyncio.Semaphore(concurrency)
    latencies_ms: list[float] = []
    status_counts: Counter[str] = Counter()

    async def request_once(client: httpx.AsyncClient) -> None:
        async with semaphore:
            started = time.perf_counter()
            try:
                response = await client.get(target)
                status_counts[str(response.status_code)] += 1
            except httpx.HTTPError as error:
                status_counts[f"error:{type(error).__name__}"] += 1
            finally:
                latencies_ms.append((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=5) as client:
        await asyncio.gather(*(request_once(client) for _ in range(requests)))
    elapsed_seconds = time.perf_counter() - started
    failures = sum(count for status_code, count in status_counts.items() if status_code != "200")
    return {
        "target": target,
        "requests": requests,
        "concurrency": concurrency,
        "elapsedSeconds": round(elapsed_seconds, 3),
        "requestsPerSecond": round(requests / elapsed_seconds, 2),
        "statusCounts": dict(sorted(status_counts.items())),
        "failures": failures,
        "latencyMs": {
            "p50": round(median(latencies_ms), 2),
            "p95": round(percentile(latencies_ms, 95), 2),
            "max": round(max(latencies_ms), 2),
        },
        "interpretation": "Local liveness/readiness probe only; not a workflow capacity result.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="http://127.0.0.1:8000/ready")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-p95-ms", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = asyncio.run(run_probe(arguments.target, arguments.requests, arguments.concurrency))
    except (ValueError, httpx.HTTPError) as error:
        print(f"Load probe failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    p95_ms = report["latencyMs"]["p95"]
    if report["failures"] or (arguments.max_p95_ms is not None and p95_ms > arguments.max_p95_ms):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Simple load testing script for the ML inference service.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --requests 100

Measures latency and failure rate for POST /predict endpoint.
"""

import argparse
import statistics
import time
from typing import List

import requests


def run_load_test(base_url: str, num_requests: int) -> None:
    url = f"{base_url}/predict"
    payload = {"data": {"feature_a": 1.0, "feature_b": 0.5, "feature_c": -1.0}}

    latencies: List[float] = []
    failures = 0

    print(f"Starting load test: {num_requests} requests to {url}")
    print("-" * 50)

    for i in range(num_requests):
        start_time = time.perf_counter()
        try:
            response = requests.post(url, json=payload, timeout=10)
            latency = time.perf_counter() - start_time
            latencies.append(latency)

            if response.status_code != 200:
                failures += 1
                print(f"Request {i+1}: FAILED ({response.status_code}) - {latency:.3f}s")
            else:
                if i < 5:  # print first few successes
                    print(f"Request {i+1}: OK - {latency:.3f}s")
                elif i == 5:
                    print("... (suppressing further success logs)")
        except Exception as e:
            latency = time.perf_counter() - start_time
            latencies.append(latency)
            failures += 1
            print(f"Request {i+1}: ERROR ({e}) - {latency:.3f}s")

    # summary
    total_time = sum(latencies)
    avg_latency = statistics.mean(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    success_rate = ((num_requests - failures) / num_requests) * 100

    print("-" * 50)
    print("Load Test Results:")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {num_requests - failures}")
    print(f"Failed: {failures}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average latency: {avg_latency:.3f}s")
    print(f"Min latency: {min_latency:.3f}s")
    print(f"Max latency: {max_latency:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Requests per second: {num_requests / total_time:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the ML inference service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests to send")
    args = parser.parse_args()

    run_load_test(args.url, args.requests)
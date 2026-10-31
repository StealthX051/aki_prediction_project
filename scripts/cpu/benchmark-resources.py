"""Bounded synthetic LR BLAS benchmark; prints JSON, never reads research data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime_resources import available_cpu_count, resource_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3800)
    parser.add_argument("--features", type=int, default=384)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blas-threads", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args()
    if min(args.samples, args.features, args.repeats, *args.blas_threads) < 1:
        parser.error("Sizes, repetitions and thread counts must be positive")
    rng = np.random.default_rng(42)
    features = np.ascontiguousarray(rng.normal(size=(args.samples, args.features)))
    logits = features[:, :min(8, args.features)].sum(axis=1) * 0.4 - 3
    labels = rng.binomial(1, 1 / (1 + np.exp(-logits)))
    runs = []
    for threads in dict.fromkeys(min(value, available_cpu_count()) for value in args.blas_threads):
        elapsed, iterations = [], []
        with threadpool_limits(limits=threads, user_api="blas"):
            for iteration in range(args.repeats + 1):
                model = LogisticRegression(C=1, solver="lbfgs", max_iter=5000)
                started = time.perf_counter()
                model.fit(features, labels)
                duration = time.perf_counter() - started
                if iteration:  # one warm-up for native-library/cache initialization
                    elapsed.append(duration)
                    iterations.append(int(model.n_iter_.max()))
        runs.append({"blas_threads": threads, "seconds": elapsed,
                     "median_seconds": statistics.median(elapsed), "iterations": iterations})
    print(json.dumps({"purpose": "synthetic CPU tuning, not manuscript performance",
                      "shape": list(features.shape), "seed": 42, "events": int(labels.sum()),
                      "resources": resource_snapshot(), "runs": runs,
                      "fastest_blas_threads": min(runs, key=lambda run: run["median_seconds"])["blas_threads"],
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}, indent=2))


if __name__ == "__main__":
    main()

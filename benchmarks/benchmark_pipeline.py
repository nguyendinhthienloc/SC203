"""Benchmark script for the IRAL pipeline.
Generates synthetic corpora of varying sizes and measures runtime and (optionally) memory usage.
Outputs a CSV summary in benchmarks/benchmark_results.csv.

Run:
  python benchmarks/benchmark_pipeline.py --sizes 10 100 500 --repeats 2
"""

from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import csv
import statistics

import pandas as pd

try:
    import psutil  # optional
except ImportError:  # pragma: no cover
    psutil = None

from src.run_pipeline import run_pipeline

BASE_VOCAB = [
    "analysis", "development", "system", "implementation", "data", "model", "research", "human", "ai", "generation",
    "nominalization", "linguistic", "approach", "method", "structure", "result", "feature", "experiment", "design", "performance"
]
SENTENCE_PATTERNS = [
    "The {w1} of the {w2} improves {w3} results.",
    "Our {w1} enables better {w2} for {w3} studies.",
    "A {w1} driven {w2} supports robust {w3} analysis.",
    "Enhanced {w1} and {w2} yield stable {w3} outcomes.",
]


def synth_document(avg_sentences: int = 5) -> str:
    sentences = []
    for _ in range(avg_sentences):
        w1, w2, w3 = random.sample(BASE_VOCAB, 3)
        pattern = random.choice(SENTENCE_PATTERNS)
        sentences.append(pattern.format(w1=w1, w2=w2, w3=w3))
    return " ".join(sentences)


def generate_corpus(n: int) -> pd.DataFrame:
    rows = []
    for i in range(n):
        label = 0 if i < n // 2 else 1
        rows.append({"text": synth_document(), "label": label})
    return pd.DataFrame(rows)


def benchmark_run(size: int, repeat: int, outdir: Path) -> dict:
    durations = []
    mem_peaks = []
    for _ in range(repeat):
        df = generate_corpus(size)
        csv_path = outdir / f"synthetic_{size}.csv"
        df.to_csv(csv_path, index=False)
        start = time.perf_counter()
        if psutil:
            process = psutil.Process()
            mem_start = process.memory_info().rss
        run_pipeline(
            input_path=str(csv_path),
            textcol="text",
            labelcol="label",
            outdir=str(outdir / f"results_{size}"),
            batch_size=64,
            n_process=1,
            skip_keywords=False,
            seed=123,
            nominalization_mode="balanced",
            collocation_min_count=3,
        )
        dur = time.perf_counter() - start
        durations.append(dur)
        if psutil:
            mem_end = process.memory_info().rss
            mem_peaks.append(mem_end - mem_start)
    return {
        "size": size,
        "repeats": repeat,
        "duration_mean": statistics.mean(durations),
        "duration_stdev": statistics.pstdev(durations),
        "duration_min": min(durations),
        "duration_max": max(durations),
        "mem_peak_mean": statistics.mean(mem_peaks) if mem_peaks else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark IRAL pipeline synthetic corpora")
    parser.add_argument("--sizes", nargs="*", type=int, default=[10, 100, 500], help="Document counts to benchmark")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per size")
    parser.add_argument("--outdir", type=str, default="benchmarks/run", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for s in args.sizes:
        print(f"Benchmarking size={s} repeats={args.repeats} ...")
        res = benchmark_run(s, args.repeats, outdir)
        results.append(res)
        print(f"  -> mean {res['duration_mean']:.3f}s (min {res['duration_min']:.3f}s, max {res['duration_max']:.3f}s)")

    csv_path = outdir / "benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()

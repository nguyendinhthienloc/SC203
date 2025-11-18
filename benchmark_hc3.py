"""Quick benchmark to estimate HC3 full dataset runtime."""
import time
import pandas as pd
from src.run_pipeline import run_pipeline

print("HC3 Medicine Dataset Benchmark")
print("="*60)

# Use medicine dataset (100 balanced docs)
print("\nBenchmarking 100 documents (HC3 Medicine)...")
start = time.perf_counter()
result = run_pipeline(
    input_path="data/raw/hc3_medicine.csv",
    textcol="text",
    labelcol="label",
    outdir="results_benchmark",
    batch_size=64,
    n_process=1,
    skip_keywords=False,  # Include keywords for realistic estimate
    seed=42,
    verbose=False
)
duration = time.perf_counter() - start

print(f"\n{'='*60}")
print(f"✅ 100 docs processed: {duration:.2f}s ({duration/100:.3f}s per doc)")
print(f"\nEstimates for full HC3 (24,322 docs):")
print(f"  Optimistic (batching helps): ~{(duration/100)*24322/60:.0f} minutes ({(duration/100)*24322/3600:.1f} hours)")
print(f"  Conservative (overhead):     ~{(duration/100)*24322*1.2/60:.0f} minutes ({(duration/100)*24322*1.2/3600:.1f} hours)")
print(f"\nRecommendation:")
if duration < 30:
    print(f"  ✅ Medicine (100 docs) should finish in <30s - try it first!")
    print(f"  ⚠️  Full HC3 will take ~{(duration/100)*24322/3600:.1f} hours - run overnight")
else:
    print(f"  ⚠️  Consider running in chunks or using batch_size=128")
print(f"{'='*60}")

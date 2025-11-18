"""Tiny end-to-end test using run_pipeline with new parameters."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_pipeline import run_pipeline


def test_end_to_end_tiny():
    df = pd.DataFrame([
        {"text": "Human development of ideas.", "label": 0},
        {"text": "AI implementation and generation.", "label": 1},
    ])
    csv_path = "tiny_input.csv"
    df.to_csv(csv_path, index=False)
    results = run_pipeline(csv_path, textcol="text", labelcol="label", outdir="results_tiny", seed=42, skip_keywords=False, batch_size=2)
    assert len(results) == 2
    assert 'word_count' in results.columns
    assert 'nominal_lemma_count' in results.columns


if __name__ == "__main__":
    test_end_to_end_tiny()
    print("Tiny end-to-end test passed")
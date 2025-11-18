"""Tests for log-odds ratio keyword extraction stability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.collocations import log_odds_ratio
from collections import Counter


def test_log_odds_basic():
    counts_a = Counter({"alpha": 10, "beta": 5, "gamma": 1})
    counts_b = Counter({"alpha": 2, "delta": 8, "gamma": 1})

    result = log_odds_ratio(counts_a, counts_b)
    assert "alpha" in {w for w, _ in result["keywords_A"]} or "alpha" in {w for w, _ in result["keywords_B"]}
    # Ensure no division by zero issues
    for _, score in result["keywords_A"] + result["keywords_B"]:
        assert score == score  # not NaN


def test_log_odds_empty():
    result = log_odds_ratio(Counter(), Counter())
    assert result["keywords_A"] == [] and result["keywords_B"] == []


if __name__ == "__main__":
    test_log_odds_basic()
    test_log_odds_empty()
    print("Log-odds tests passed")
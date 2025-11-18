"""Edge case tests for statistical functions."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stats_analysis import welch_ttest, mann_whitney, cohen_d, mean_diff_CI


def test_empty_inputs():
    w = welch_ttest([], [1,2,3])
    assert np.isnan(w['t_statistic']) and np.isnan(w['p_value'])
    m = mann_whitney([], [1,2])
    assert np.isnan(m['u_statistic'])
    assert m['p_value'] == np.nan or m['p_value'] == np.nan
    d = cohen_d([], [1,2])
    assert d == 0.0
    ci = mean_diff_CI([], [1,2])
    assert np.isnan(ci['mean_diff'])


def test_identical_values():
    a = [1,1,1]
    b = [1,1,1]
    w = welch_ttest(a,b)
    assert np.isnan(w['t_statistic']) or w['p_value'] >= 0.5
    d = cohen_d(a,b)
    assert d == 0.0


if __name__ == "__main__":
    test_empty_inputs()
    test_identical_values()
    print("Statistical edge case tests passed")
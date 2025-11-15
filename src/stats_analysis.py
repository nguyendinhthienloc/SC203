"""
Statistical analysis module.

Implements:
- Welch's t-test
- Mann-Whitney U test
- Cohen's d effect size
- Confidence intervals
"""

import numpy as np
from scipy import stats


def cohen_d(a, b):
    """
    Compute Cohen's d effect size.
    
    Cohen's d = (mean_a - mean_b) / pooled_std
    
    Parameters
    ----------
    a : array-like
        Sample A
    b : array-like
        Sample B
    
    Returns
    -------
    float
        Cohen's d effect size
    """
    a = np.array(a)
    b = np.array(b)
    
    if len(a) == 0 or len(b) == 0:
        return 0.0
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    # Pooled standard deviation
    n_a = len(a)
    n_b = len(b)
    
    var_a = np.var(a, ddof=1) if n_a > 1 else 0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0
    
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean_a - mean_b) / pooled_std
    
    return d


def welch_ttest(a, b):
    """
    Perform Welch's t-test (unequal variances t-test).
    
    Parameters
    ----------
    a : array-like
        Sample A
    b : array-like
        Sample B
    
    Returns
    -------
    dict
        Dictionary containing:
        - t_statistic: t-value
        - p_value: two-tailed p-value
        - mean_a: mean of sample A
        - mean_b: mean of sample B
        - std_a: standard deviation of sample A
        - std_b: standard deviation of sample B
    """
    a = np.array(a)
    b = np.array(b)
    
    if len(a) < 2 or len(b) < 2:
        return {
            't_statistic': np.nan,
            'p_value': np.nan,
            'mean_a': np.mean(a) if len(a) > 0 else np.nan,
            'mean_b': np.mean(b) if len(b) > 0 else np.nan,
            'std_a': np.nan,
            'std_b': np.nan
        }
    
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    
    return {
        't_statistic': t_stat,
        'p_value': p_val,
        'mean_a': np.mean(a),
        'mean_b': np.mean(b),
        'std_a': np.std(a, ddof=1),
        'std_b': np.std(b, ddof=1)
    }


def mann_whitney(a, b):
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Parameters
    ----------
    a : array-like
        Sample A
    b : array-like
        Sample B
    
    Returns
    -------
    dict
        Dictionary containing:
        - u_statistic: U-value
        - p_value: two-tailed p-value
        - median_a: median of sample A
        - median_b: median of sample B
    """
    a = np.array(a)
    b = np.array(b)
    
    if len(a) == 0 or len(b) == 0:
        return {
            'u_statistic': np.nan,
            'p_value': np.nan,
            'median_a': np.nan,
            'median_b': np.nan
        }
    
    try:
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative='two-sided')
    except ValueError:
        # Handle cases where all values are identical
        return {
            'u_statistic': np.nan,
            'p_value': 1.0,
            'median_a': np.median(a),
            'median_b': np.median(b)
        }
    
    return {
        'u_statistic': u_stat,
        'p_value': p_val,
        'median_a': np.median(a),
        'median_b': np.median(b)
    }


def mean_diff_CI(a, b, alpha=0.05):
    """
    Compute confidence interval for difference in means.
    
    Uses Welch's method for unequal variances.
    
    Parameters
    ----------
    a : array-like
        Sample A
    b : array-like
        Sample B
    alpha : float, default=0.05
        Significance level (for 95% CI, use 0.05)
    
    Returns
    -------
    dict
        Dictionary containing:
        - mean_diff: difference in means (a - b)
        - ci_lower: lower bound of confidence interval
        - ci_upper: upper bound of confidence interval
        - confidence_level: confidence level (e.g., 0.95)
    """
    a = np.array(a)
    b = np.array(b)
    
    if len(a) < 2 or len(b) < 2:
        return {
            'mean_diff': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'confidence_level': 1 - alpha
        }
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    mean_diff = mean_a - mean_b
    
    # Standard errors
    se_a = np.std(a, ddof=1) / np.sqrt(len(a))
    se_b = np.std(b, ddof=1) / np.sqrt(len(b))
    se_diff = np.sqrt(se_a**2 + se_b**2)
    
    # Welch-Satterthwaite degrees of freedom
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    n_a = len(a)
    n_b = len(b)
    
    df = ((var_a / n_a + var_b / n_b)**2) / \
         ((var_a / n_a)**2 / (n_a - 1) + (var_b / n_b)**2 / (n_b - 1))
    
    # Critical value from t-distribution
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': 1 - alpha
    }


def compare_groups(group_a, group_b, metric_name="metric"):
    """
    Comprehensive statistical comparison between two groups.
    
    Parameters
    ----------
    group_a : array-like
        Values for group A
    group_b : array-like
        Values for group B
    metric_name : str, default="metric"
        Name of the metric being compared
    
    Returns
    -------
    dict
        Complete statistical comparison results
    """
    results = {
        'metric': metric_name,
        'welch_ttest': welch_ttest(group_a, group_b),
        'mann_whitney': mann_whitney(group_a, group_b),
        'cohen_d': cohen_d(group_a, group_b),
        'ci_95': mean_diff_CI(group_a, group_b, alpha=0.05)
    }
    
    return results

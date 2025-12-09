"""Statistical analysis helpers used in the IRAL replication.

Implemented formulas:
- **Welch's t-test**: t = (\bar{x}-\bar{y}) / sqrt(s_x^2/n_x + s_y^2/n_y)
- **Mann-Whitney U** per SciPy (non-parametric location test)
- **Cohen's d**: d = (\bar{x}-\bar{y}) / s_pooled
- **Mean difference CI** (Welch) using t_{1-\alpha/2, df}
- **FDR-BH correction** for multiple comparisons
"""

import numpy as np
from scipy import stats
import random


def set_random_seed(seed: int):
    """Set random seeds for reproducibility across numpy and random.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


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
    
    if n_a + n_b <= 2:
        return 0.0

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


def adjust_pvalues(p_values, method="fdr_bh"):
    """Adjust a list of p-values for multiple comparisons.

    Parameters
    ----------
    p_values : list of float
        Raw p-values in the order they were computed.
    method : str, default="fdr_bh"
        Currently only Benjamini-Hochberg FDR is implemented.

    Returns
    -------
    list of float
        Adjusted p-values (NaN preserved for missing inputs).
    """
    if method != "fdr_bh":
        raise ValueError("Only Benjamini-Hochberg correction is supported")

    indexed = [(idx, p) for idx, p in enumerate(p_values) if not np.isnan(p)]
    m = len(indexed)
    adjusted = [np.nan] * len(p_values)

    if m == 0:
        return adjusted

    # Sort by p-value ascending
    ranked = sorted(indexed, key=lambda x: x[1])

    # Benjamini-Hochberg step-up procedure
    prev = 1.0
    for rank, (idx, p) in enumerate(reversed(ranked), start=1):
        bh_value = min(prev, (p * m) / (m - rank + 1))
        prev = bh_value
        adjusted[idx] = bh_value

    return adjusted


def gecs_classification_metrics(rouge2_scores, labels, threshold=0.924):
    """
    Compute classification metrics for GECS scores.
    
    Using Rouge-2 score as a binary classifier:
    - Score >= threshold → Predicted as AI (1)
    - Score < threshold → Predicted as Human (0)
    
    Parameters
    ----------
    rouge2_scores : array-like
        Rouge-2 F-scores for all texts
    labels : array-like
        True binary labels (0=human, 1=AI)
    threshold : float, default=0.924
        Classification threshold (scores >= threshold are classified as AI)
    
    Returns
    -------
    dict
        Classification metrics including:
        - accuracy, precision, recall, f1_score
        - confusion_matrix: [[TN, FP], [FN, TP]]
        - threshold: the threshold used
    """
    rouge2_scores = np.array(rouge2_scores)
    labels = np.array(labels)
    
    # Filter out None/NaN values
    valid_mask = ~np.isnan(rouge2_scores)
    rouge2_scores = rouge2_scores[valid_mask]
    labels = labels[valid_mask]
    
    if len(rouge2_scores) == 0:
        return {
            'accuracy': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'confusion_matrix': [[np.nan, np.nan], [np.nan, np.nan]],
            'threshold': threshold,
            'n_samples': 0
        }
    
    # Make predictions
    predictions = (rouge2_scores >= threshold).astype(int)
    
    # Compute metrics
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, confusion_matrix
    )
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    cm = confusion_matrix(labels, predictions).tolist()
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm,
        'threshold': threshold,
        'n_samples': len(rouge2_scores)
    }

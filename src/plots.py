"""
Visualization module for generating plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Set style
sns.set_style("whitegrid")
sns.set_context("notebook")


def boxplot_by_label(df, label_col, metric_col, outpath, 
                     label_names=None, title=None):
    """
    Create boxplot comparing metric across labels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing metrics
    label_col : str
        Column name for labels (grouping variable)
    metric_col : str
        Column name for metric to plot
    outpath : str
        Output file path for figure
    label_names : dict, optional
        Mapping of label values to display names
    title : str, optional
        Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Prepare data
    if label_names:
        df_plot = df.copy()
        df_plot[label_col] = df_plot[label_col].map(label_names)
    else:
        df_plot = df
    
    # Create boxplot
    sns.boxplot(data=df_plot, x=label_col, y=metric_col, palette="Set2")
    
    # Add title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'{metric_col} by {label_col}', fontsize=14, fontweight='bold')
    
    plt.xlabel(label_col.capitalize(), fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def bar_with_ci(df, label_col, metric_col, outpath,
                label_names=None, title=None, alpha=0.05):
    """
    Create bar plot with confidence intervals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing metrics
    label_col : str
        Column name for labels (grouping variable)
    metric_col : str
        Column name for metric to plot
    outpath : str
        Output file path for figure
    label_names : dict, optional
        Mapping of label values to display names
    title : str, optional
        Plot title
    alpha : float, default=0.05
        Significance level for confidence intervals
    """
    plt.figure(figsize=(8, 6))
    
    # Calculate means and confidence intervals
    groups = df.groupby(label_col)[metric_col]
    
    means = groups.mean()
    stds = groups.std()
    counts = groups.count()
    
    # Calculate confidence intervals
    ci = {}
    for label in means.index:
        data = df[df[label_col] == label][metric_col].values
        if len(data) > 1:
            ci_val = stats.t.ppf(1 - alpha/2, len(data) - 1) * stds[label] / np.sqrt(counts[label])
        else:
            ci_val = 0
        ci[label] = ci_val
    
    # Prepare labels
    if label_names:
        plot_labels = [label_names.get(label, str(label)) for label in means.index]
    else:
        plot_labels = [str(label) for label in means.index]
    
    # Create bar plot
    x_pos = np.arange(len(means))
    colors = sns.color_palette("Set2", len(means))
    
    bars = plt.bar(x_pos, means.values, yerr=[ci[label] for label in means.index],
                   capsize=10, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, means.values)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(x_pos, plot_labels)
    
    # Add title
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    else:
        plt.title(f'{metric_col} by {label_col}', fontsize=14, fontweight='bold')
    
    plt.xlabel(label_col.capitalize(), fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def keyword_barplot(keywords, outpath, title="Top Keywords", top_n=20):
    """
    Create horizontal bar plot of keywords with scores.
    
    Parameters
    ----------
    keywords : list of tuple
        List of (word, score) tuples
    outpath : str
        Output file path for figure
    title : str, default="Top Keywords"
        Plot title
    top_n : int, default=20
        Number of top keywords to display
    """
    if not keywords:
        print(f"No keywords to plot for {outpath}")
        return
    
    # Take top N
    keywords = keywords[:top_n]
    words, scores = zip(*keywords)
    
    plt.figure(figsize=(10, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(words))
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores]
    
    plt.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.5)
    plt.yticks(y_pos, words)
    plt.xlabel('Log-Odds Ratio', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Add vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def create_comparison_plot(df, label_col, metrics, outdir, label_names=None):
    """
    Create multiple comparison plots for a list of metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data containing metrics
    label_col : str
        Column name for labels
    metrics : list of str
        List of metric column names to plot
    outdir : str
        Output directory for figures
    label_names : dict, optional
        Mapping of label values to display names
    """
    os.makedirs(outdir, exist_ok=True)
    
    for metric in metrics:
        if metric not in df.columns:
            continue
        
        # Boxplot
        boxplot_path = os.path.join(outdir, f'{metric}_boxplot.png')
        boxplot_by_label(df, label_col, metric, boxplot_path, label_names)
        
        # Bar plot with CI
        bar_path = os.path.join(outdir, f'{metric}_barplot.png')
        bar_with_ci(df, label_col, metric, bar_path, label_names)

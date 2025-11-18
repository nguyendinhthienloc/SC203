"""
Visualization module for generating IRAL-style publication-quality plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# Set IRAL journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

sns.set_style("whitegrid", {
    'grid.linestyle': ':',
    'grid.linewidth': 0.5,
    'grid.color': '#CCCCCC'
})
sns.set_palette("colorblind")


def boxplot_by_label(df, label_col, metric_col, outpath, 
                     label_names=None, title=None):
    """
    Create IRAL-style boxplot comparing metric across labels.
    
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
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
    # Prepare data
    if label_names:
        df_plot = df.copy()
        df_plot[label_col] = df_plot[label_col].map(label_names)
    else:
        df_plot = df
    
    # Create boxplot with IRAL styling
    box_colors = ['#E8E8E8', '#B8B8B8']  # Light gray shades for professional look
    bp = sns.boxplot(
        data=df_plot, 
        x=label_col, 
        y=metric_col, 
        palette=box_colors,
        width=0.5,
        linewidth=1.5,
        fliersize=4,
        ax=ax
    )
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Format labels
    ylabel = _format_metric_name(metric_col)
    ax.set_xlabel('')  # Remove x-axis label for cleaner look
    ax.set_ylabel(ylabel, fontweight='normal')
    
    # Add subtle title if provided (IRAL style: minimal titles)
    if title:
        ax.set_title(title, pad=10, fontweight='normal', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def bar_with_ci(df, label_col, metric_col, outpath,
                label_names=None, title=None, alpha=0.05):
    """
    Create IRAL-style bar plot with 95% confidence intervals.
    
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
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    
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
    
    # Create bar plot with IRAL styling
    x_pos = np.arange(len(means))
    bar_colors = ['#D0D0D0', '#909090']  # Professional gray scale
    
    bars = ax.bar(
        x_pos, 
        means.values, 
        yerr=[ci[label] for label in means.index],
        capsize=5,
        color=bar_colors,
        edgecolor='black',
        linewidth=1.0,
        error_kw={'linewidth': 1.5, 'ecolor': 'black'}
    )
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Format labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_labels)
    ax.set_xlabel('')  # Remove x-axis label for cleaner look
    
    ylabel = _format_metric_name(metric_col)
    ax.set_ylabel(ylabel, fontweight='normal')
    
    # Add value labels on bars (more subtle)
    for i, (bar, val, ci_val) in enumerate(zip(bars, means.values, [ci[label] for label in means.index])):
        height = bar.get_height()
        # Position label above error bar
        label_height = height + ci_val + ax.get_ylim()[1] * 0.02
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            label_height,
            f'{val:.2f}',
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='normal'
        )
    
    # Add subtle title if provided
    if title:
        ax.set_title(title, pad=10, fontweight='normal', fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def keyword_barplot(keywords, outpath, title="Top Keywords", top_n=20):
    """
    Create IRAL-style horizontal bar plot of keywords with log-odds scores.
    
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
    
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    
    # Create horizontal bar plot with IRAL styling
    y_pos = np.arange(len(words))
    
    # Professional grayscale coloring
    colors = ['#606060' if s > 0 else '#A0A0A0' for s in scores]
    
    bars = ax.barh(
        y_pos, 
        scores, 
        color=colors, 
        edgecolor='black', 
        linewidth=0.8
    )
    
    # Customize appearance
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words, fontsize=10)
    ax.set_xlabel('Log-Odds Ratio', fontweight='normal', fontsize=11)
    ax.set_title(title, pad=10, fontweight='normal', fontsize=11)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add vertical reference line at x=0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.7)
    
    # Add subtle grid
    ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Invert y-axis so highest scores are at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()


def create_comparison_plot(df, label_col, metrics, outdir, label_names=None):
    """
    Create multiple IRAL-style comparison plots for a list of metrics.
    
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


def _format_metric_name(metric_col):
    """
    Format metric name for publication-quality display.
    
    Parameters
    ----------
    metric_col : str
        Raw metric column name (e.g., 'noun_ratio')
    
    Returns
    -------
    str
        Formatted metric name (e.g., 'Noun Ratio')
    """
    # Special cases for academic terminology
    replacements = {
        'type_token_ratio': 'Type-Token Ratio (TTR)',
        'ttr': 'Type-Token Ratio (TTR)',
        'avg_sentence_len': 'Average Sentence Length',
        'avg_word_len': 'Average Word Length',
        'word_count': 'Word Count',
        'sentence_count': 'Sentence Count',
        'noun_ratio': 'Noun Ratio',
        'verb_ratio': 'Verb Ratio',
        'adj_ratio': 'Adjective Ratio',
        'adv_ratio': 'Adverb Ratio',
        'nominal_lemma_ratio': 'Nominalization Ratio',
        'nominal_ratio': 'Nominalization Ratio',
        'noun_count': 'Noun Count',
        'verb_count': 'Verb Count',
        'nominal_suffix_count': 'Nominal Suffix Count'
    }
    
    if metric_col in replacements:
        return replacements[metric_col]
    
    # Default: title case with underscores replaced
    return metric_col.replace('_', ' ').title()

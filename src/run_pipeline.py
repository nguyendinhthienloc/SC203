"""
Main pipeline orchestrator for IRAL text analysis.

This module ties together all analysis steps:
1. Ingest data
2. Clean text
3. Tokenize and POS tag
4. Compute features
5. Detect nominalizations
6. Extract collocations
7. Identify keywords
8. Statistical testing
9. Generate visualizations
10. Export results
"""

import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from .ingest import ingest
from .clean import clean_text
from .pos_tools import tokenize_and_pos
from .features import compute_basic_metrics, compute_pos_features
from .nominalization import analyze_nominalization
from .collocations import extract_collocations, extract_keywords
from .stats_analysis import compare_groups, adjust_pvalues
from .plots import create_comparison_plot, keyword_barplot
from .plots_iral import create_three_iral_figures, cleanup_old_figures


def run_pipeline(input_path, textcol="text", labelcol="label", outdir="results"):
    """
    Execute complete IRAL analysis pipeline.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file or folder of text files
    textcol : str, default="text"
        Column name for text content (CSV only)
    labelcol : str, default="label"
        Column name for labels (CSV only)
    outdir : str, default="results"
        Output directory for results
    
    Returns
    -------
    pd.DataFrame
        Augmented dataframe with all computed features
    """
    print("=" * 80)
    print("IRAL Text Analysis Pipeline")
    print("=" * 80)
    
    # Step 1: Ingest data
    print("\n[1/10] Ingesting data...")
    df = ingest(input_path, textcol=textcol, labelcol=labelcol)
    print(f"  Loaded {len(df)} documents")
    print(f"  Label distribution: {df[labelcol].value_counts().to_dict()}")
    
    # Step 2: Clean text
    print("\n[2/10] Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Step 3-6: Process each document
    print("\n[3/10] Processing documents...")
    
    results_list = []
    docs_list = []
    tokens_list = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        text = row['cleaned_text']
        
        # Tokenize and POS tag
        pos_result = tokenize_and_pos(text)
        
        tokens = pos_result['words']
        sentences = pos_result['sentences']
        pos_counts = pos_result['pos_counts']
        pos_tokens = pos_result.get('pos_tokens')
        doc = pos_result['doc']
        
        # Store for later use
        tokens_list.append(tokens)
        docs_list.append(doc)
        
        # Compute basic metrics
        basic_metrics = compute_basic_metrics(tokens, sentences)
        
        # Compute POS features
        pos_features = compute_pos_features(pos_counts, basic_metrics['word_count'])
        
        # Nominalization analysis
        nominal_results = analyze_nominalization(doc=doc, tokens=tokens, pos_tokens=pos_tokens)
        
        # Extract collocations (top 10 per document)
        collocations = extract_collocations(tokens, top_n=10)
        
        # Combine all features
        result = {
            'id': row['id'],
            'label': row[labelcol],
            **basic_metrics,
            **pos_features
        }
        
        # Add nominalization features
        if nominal_results['lemma_based']:
            result['nominal_lemma_count'] = nominal_results['lemma_based']['nominal_from_verb']
            result['nominal_lemma_ratio'] = nominal_results['lemma_based']['nominal_ratio']
        else:
            result['nominal_lemma_count'] = 0
            result['nominal_lemma_ratio'] = 0.0
        
        if nominal_results['suffix_based']:
            result['nominal_suffix_count'] = nominal_results['suffix_based']['nominal_from_verb']
        else:
            result['nominal_suffix_count'] = 0
        
        # Store top collocation
        if collocations['top_collocations']:
            top_coll = collocations['top_collocations'][0]
            result['top_collocation'] = f"{top_coll[0][0]} {top_coll[0][1]}"
            result['top_collocation_pmi'] = top_coll[1]
        else:
            result['top_collocation'] = ""
            result['top_collocation_pmi'] = 0.0
        
        results_list.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    # Step 7: Keywords analysis (across groups)
    print("\n[4/10] Extracting keywords...")
    if labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
        # Split by label
        labels = sorted(results_df[labelcol].unique())
        label_0_indices = results_df[results_df[labelcol] == labels[0]].index
        label_1_indices = results_df[results_df[labelcol] == labels[1]].index
        
        tokens_0 = [token for idx in label_0_indices for token in tokens_list[idx]]
        tokens_1 = [token for idx in label_1_indices for token in tokens_list[idx]]
        
        # Use lower min_freq for small datasets
        min_freq = 2 if len(results_df) < 10 else 5
        keywords = extract_keywords(tokens_0, tokens_1, min_freq=min_freq)
        
        print(f"  Extracted {len(keywords['keywords_A'])} keywords for group {labels[0]}")
        print(f"  Extracted {len(keywords['keywords_B'])} keywords for group {labels[1]}")
    else:
        keywords = None
        print("  Skipping keyword extraction (requires exactly 2 groups)")
    
    # Step 8: Statistical analysis
    print("\n[5/10] Running statistical tests...")
    
    if labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
        labels = sorted(results_df[labelcol].unique())
        
        metrics_to_test = [
            'word_count', 'sentence_count', 'avg_sentence_len', 
            'avg_word_len', 'type_token_ratio',
            'noun_count', 'verb_count', 'noun_ratio', 'verb_ratio',
            'nominal_lemma_count', 'nominal_lemma_ratio', 'nominal_suffix_count'
        ]
        
        stats_results = []
        
        for metric in metrics_to_test:
            if metric not in results_df.columns:
                continue
            
            group_0 = results_df[results_df[labelcol] == labels[0]][metric].values
            group_1 = results_df[results_df[labelcol] == labels[1]][metric].values
            
            comparison = compare_groups(group_0, group_1, metric_name=metric)
            stats_results.append(comparison)
        
        if stats_results:
            welch_adj = adjust_pvalues([r['welch_ttest']['p_value'] for r in stats_results])
            mann_adj = adjust_pvalues([r['mann_whitney']['p_value'] for r in stats_results])

            for result, w_adj, m_adj in zip(stats_results, welch_adj, mann_adj):
                result['welch_ttest']['p_value_adj'] = w_adj
                result['mann_whitney']['p_value_adj'] = m_adj

        print(f"  Completed statistical tests for {len(stats_results)} metrics")
    else:
        stats_results = []
        print("  Skipping statistical tests (requires exactly 2 groups)")
    
    # Step 9: Create visualizations (IRAL 3-figure style)
    print("\n[6/10] Creating IRAL-style visualizations...")
    
    figures_dir = os.path.join(outdir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    if labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
        labels = sorted(results_df[labelcol].unique())
        label_names = {labels[0]: f"Group {labels[0]}", labels[1]: f"Group {labels[1]}"}
        
        # Customize label names if possible
        if labels[0] == 0 and labels[1] == 1:
            label_names = {0: "Human", 1: "AI"}
        
        # Create the 3 main IRAL figures
        if keywords:
            create_three_iral_figures(
                keywords['keywords_A'],
                keywords['keywords_B'],
                figures_dir,
                label_names=label_names
            )
            
            # Clean up old individual metric plots
            cleanup_old_figures(figures_dir)
        else:
            print("  ⚠ Skipping keyword figures (no keywords extracted)")
        
        print(f"  ✓ Created 3 IRAL figures in {figures_dir}")
    else:
        print("  Skipping visualizations (requires exactly 2 groups)")
    
    # Step 10: Export results
    print("\n[7/10] Exporting results...")
    
    # Export augmented CSV
    output_csv = os.path.join(outdir, "human_vs_ai_augmented.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"  Saved augmented data to {output_csv}")
    
    # Export statistical results
    if stats_results:
        tables_dir = os.path.join(outdir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        
        stats_summary = []
        for result in stats_results:
            stats_summary.append({
                'metric': result['metric'],
                'mean_group_0': result['welch_ttest']['mean_a'],
                'mean_group_1': result['welch_ttest']['mean_b'],
                'std_group_0': result['welch_ttest']['std_a'],
                'std_group_1': result['welch_ttest']['std_b'],
                't_statistic': result['welch_ttest']['t_statistic'],
                'p_value': result['welch_ttest']['p_value'],
                'p_value_adj': result['welch_ttest'].get('p_value_adj'),
                'cohen_d': result['cohen_d'],
                'u_statistic': result['mann_whitney']['u_statistic'],
                'mw_p_value': result['mann_whitney']['p_value'],
                'mw_p_value_adj': result['mann_whitney'].get('p_value_adj'),
                'mean_diff': result['ci_95']['mean_diff'],
                'ci_lower': result['ci_95']['ci_lower'],
                'ci_upper': result['ci_95']['ci_upper']
            })
        
        stats_df = pd.DataFrame(stats_summary)
        stats_csv = os.path.join(tables_dir, "statistical_tests.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved statistical tests to {stats_csv}")
    
    # Export keywords
    if keywords:
        tables_dir = os.path.join(outdir, "tables")
        
        keywords_0_df = pd.DataFrame(keywords['keywords_A'], columns=['word', 'log_odds'])
        keywords_0_csv = os.path.join(tables_dir, "keywords_group_0.csv")
        keywords_0_df.to_csv(keywords_0_csv, index=False)
        
        keywords_1_df = pd.DataFrame(keywords['keywords_B'], columns=['word', 'log_odds'])
        keywords_1_csv = os.path.join(tables_dir, "keywords_group_1.csv")
        keywords_1_df.to_csv(keywords_1_csv, index=False)
        
        print(f"  Saved keywords to {tables_dir}")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)
    print(f"\nResults saved to: {outdir}")
    print(f"  - Augmented CSV: {output_csv}")
    print(f"  - Figures: {figures_dir}")
    if stats_results or keywords:
        print(f"  - Tables: {os.path.join(outdir, 'tables')}")
    
    return results_df

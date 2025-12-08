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
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from .ingest import ingest, validate_inputs
from .clean import clean_text
from .pos_tools import tokenize_and_pos, tokenize_and_pos_pipe
from .features import compute_basic_metrics, compute_pos_features
from .nominalization import analyze_nominalization
from .collocations import extract_collocations, extract_keywords
from .stats_analysis import compare_groups, adjust_pvalues, set_random_seed
from .plots import create_comparison_plot, keyword_barplot
from .plots_iral import create_three_iral_figures, cleanup_old_figures
from .gec_score import compute_gecs_features_batch, gecs_statistical_summary


def run_pipeline(input_path, textcol="text", labelcol="label", outdir="results", nominalization_mode: str = "balanced",
                 collocation_min_count: int = 5, skip_keywords: bool = False, min_freq_keywords: int = None,
                 batch_size: int = 32, n_process: int = 1, seed: int = None, verbose: bool = True, debug: bool = False,
                 enable_gecs: bool = False, gecs_model: str = "gpt-4o-mini"):
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
    enable_gecs : bool, default=False
        Enable Grammar Error Correction Score (GECS) features
        Requires OpenAI API and adds GEC Rouge-2 scores
    gecs_model : str, default="gpt-4o-mini"
        OpenAI model to use for grammar correction
    
    Returns
    -------
    pd.DataFrame
        Augmented dataframe with all computed features
    """
    # Logging setup
    log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("iral_pipeline")

    logger.info("IRAL Text Analysis Pipeline starting")
    if seed is not None:
        set_random_seed(seed)
        logger.info(f"Deterministic mode enabled with seed={seed}")
    
    # Step 1: Ingest data
    logger.info("[1/10] Ingesting data")
    df = ingest(input_path, textcol=textcol, labelcol=labelcol)
    df = validate_inputs(df, textcol=textcol, labelcol=labelcol)
    logger.info(f"Loaded {len(df)} documents")
    logger.info(f"Label distribution: {df[labelcol].value_counts().to_dict()}")
    
    # Step 2: Clean text
    logger.info("[2/10] Cleaning text")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Step 3-6: Process documents with batched spaCy pipe
    logger.info("[3/10] Processing documents (batched)")

    texts = df['cleaned_text'].tolist()
    pos_results_batch = tokenize_and_pos_pipe(texts, batch_size=batch_size, n_process=n_process)

    results_list = []
    docs_list = []
    tokens_list = []

    for (idx, row), pos_result in zip(df.iterrows(), pos_results_batch):
        tokens = pos_result['words']
        sentences = pos_result['sentences']
        pos_counts = pos_result['pos_counts']
        pos_tokens = pos_result.get('pos_tokens')
        doc = pos_result['doc']

        tokens_list.append(tokens)
        docs_list.append(doc)

        basic_metrics = compute_basic_metrics(tokens, sentences)
        pos_features = compute_pos_features(pos_counts, basic_metrics['word_count'])
        nominal_results = analyze_nominalization(doc=doc, tokens=tokens, pos_tokens=pos_tokens, mode=nominalization_mode)
        collocations = extract_collocations(tokens, top_n=10, min_count=collocation_min_count)

        result = {
            'id': row['id'],
            'label': row[labelcol],
            **basic_metrics,
            **pos_features
        }

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
    
    # Step 6.5: GECS Analysis (if enabled)
    if enable_gecs:
        logger.info("[4/10] Computing GECS features (Grammar Error Correction)")
        logger.info("This may take several minutes depending on dataset size...")
        
        gecs_results = compute_gecs_features_batch(
            df['cleaned_text'].tolist(),
            model=gecs_model,
            verbose=verbose
        )
        
        results_df['gec_text'] = [r['gec_text'] for r in gecs_results]
        results_df['gec_rouge2_score'] = [r['gec_rouge2_score'] for r in gecs_results]
        
        # Log GECS statistics
        valid_scores = [s for s in results_df['gec_rouge2_score'] if s is not None]
        if valid_scores:
            logger.info(f"GECS computed for {len(valid_scores)}/{len(results_df)} documents")
            logger.info(f"Mean Rouge-2 score: {np.mean(valid_scores):.4f} (std: {np.std(valid_scores):.4f})")
            
            if labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
                gecs_stats = gecs_statistical_summary(
                    results_df['gec_rouge2_score'].fillna(np.nan).tolist(),
                    results_df[labelcol].tolist()
                )
                logger.info(f"GECS by group:")
                logger.info(f"  Human: {gecs_stats['human_mean']:.4f} ± {gecs_stats['human_std']:.4f}")
                logger.info(f"  AI:    {gecs_stats['ai_mean']:.4f} ± {gecs_stats['ai_std']:.4f}")
                logger.info(f"  Difference: {gecs_stats['difference']:.4f} (Cohen's d: {gecs_stats['effect_size']:.4f})")
        else:
            logger.warning("GECS computation failed for all documents")
    else:
        logger.info("[4/10] GECS features disabled (use enable_gecs=True to enable)")
    
    # Step 7: Keywords analysis (across groups)
    logger.info("[5/10] Extracting keywords")
    if not skip_keywords and labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
        # Split by label
        labels = sorted(results_df[labelcol].unique())
        label_0_indices = results_df[results_df[labelcol] == labels[0]].index
        label_1_indices = results_df[results_df[labelcol] == labels[1]].index
        
        tokens_0 = [token for idx in label_0_indices for token in tokens_list[idx]]
        tokens_1 = [token for idx in label_1_indices for token in tokens_list[idx]]
        
        # Use lower min_freq for small datasets
        default_min_freq = 2 if len(results_df) < 10 else 5
        min_freq = min_freq_keywords if min_freq_keywords is not None else default_min_freq
        keywords = extract_keywords(tokens_0, tokens_1, min_freq=min_freq)
        
        logger.info(f"Extracted {len(keywords['keywords_A'])} keywords for group {labels[0]}")
        logger.info(f"Extracted {len(keywords['keywords_B'])} keywords for group {labels[1]}")
    else:
        keywords = None
        if skip_keywords:
            logger.warning("Skipping keyword extraction (--skip-keywords)")
        else:
            logger.warning("Skipping keyword extraction (requires exactly 2 groups)")
    
    # Step 8: Statistical analysis
    logger.info("[6/10] Running statistical tests")
    
    if labelcol in results_df.columns and results_df[labelcol].nunique() == 2:
        labels = sorted(results_df[labelcol].unique())
        
        metrics_to_test = [
            'word_count', 'sentence_count', 'avg_sentence_len', 
            'avg_word_len', 'type_token_ratio',
            'noun_count', 'verb_count', 'noun_ratio', 'verb_ratio',
            'nominal_lemma_count', 'nominal_lemma_ratio', 'nominal_suffix_count'
        ]
        
        # Add GECS score if enabled
        if enable_gecs and 'gec_rouge2_score' in results_df.columns:
            metrics_to_test.append('gec_rouge2_score')
        
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

        logger.info(f"Completed statistical tests for {len(stats_results)} metrics")
    else:
        stats_results = []
        print("  Skipping statistical tests (requires exactly 2 groups)")
    
    # Step 9: Create visualizations (IRAL 3-figure style)
    logger.info("[7/10] Creating IRAL-style visualizations")
    
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
            logger.warning("Skipping keyword figures (no keywords extracted)")
        
        logger.info(f"Created IRAL figures in {figures_dir}")
    else:
        logger.warning("Skipping visualizations (requires exactly 2 groups)")
    
    # Step 10: Export results
    logger.info("[8/10] Exporting results")
    
    # Export augmented CSV
    output_csv = os.path.join(outdir, "human_vs_ai_augmented.csv")
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Saved augmented data to {output_csv}")
    
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
        logger.info(f"Saved statistical tests to {stats_csv}")
    
    # Export keywords
    if keywords:
        tables_dir = os.path.join(outdir, "tables")
        
        keywords_0_df = pd.DataFrame(keywords['keywords_A'], columns=['word', 'log_odds'])
        keywords_0_csv = os.path.join(tables_dir, "keywords_group_0.csv")
        keywords_0_df.to_csv(keywords_0_csv, index=False)
        
        keywords_1_df = pd.DataFrame(keywords['keywords_B'], columns=['word', 'log_odds'])
        keywords_1_csv = os.path.join(tables_dir, "keywords_group_1.csv")
        keywords_1_df.to_csv(keywords_1_csv, index=False)
        
        logger.info(f"Saved keywords to {tables_dir}")
    
    logger.info("Pipeline completed successfully")
    logger.info(f"Results saved to: {outdir}")
    logger.info(f"Augmented CSV: {output_csv}")
    logger.info(f"Figures: {figures_dir}")
    if stats_results or keywords:
        logger.info(f"Tables: {os.path.join(outdir, 'tables')}")
    
    return results_df

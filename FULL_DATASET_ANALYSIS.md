# Full Dataset Analysis - HC3 Corpus

## Overview

This document explains the shift from small samples (100 texts) to **full dataset analysis** for more robust and reliable results.

## Why Full Datasets Matter

### Previous Approach (Limited)
- Finance: 100 human + 100 AI = **200 samples**
- Medicine: 50 human + 50 AI = **100 samples**
- **Issue**: Small samples may not capture true linguistic patterns
- **Issue**: Limited statistical power and generalizability

### New Approach (Comprehensive)
- Finance: 3,356 human + 4,384 AI = **7,740 samples**
- Medicine: 953 human + 1,301 AI = **2,254 samples**
- Open QA: 139 human + 3,253 AI = **3,392 samples**
- Reddit ELI5: 35,434 human + 16,594 AI = **52,028 samples**
- Wiki CS/AI: 789 human + 833 AI = **1,622 samples**

**Total: 67,036 samples** across 5 genres

## Benefits of Full Dataset Analysis

### 1. Statistical Robustness
- ✅ Larger sample sizes → more reliable effect sizes
- ✅ Better power to detect real differences
- ✅ Reduced risk of Type I and Type II errors
- ✅ More accurate confidence intervals

### 2. Genre Diversity
- ✅ Multiple domains (finance, medicine, QA, social media, academic)
- ✅ Different writing styles and contexts
- ✅ Varied audience types (professionals, general public, students)
- ✅ Cross-genre comparison capabilities

### 3. Imbalance Analysis
Some genres have natural imbalances:
- **Open QA**: 139 human vs 3,253 AI (AI-heavy)
- **Reddit ELI5**: 35,434 human vs 16,594 AI (human-heavy)

This reflects real-world data collection challenges and allows us to test robustness of our metrics under imbalanced conditions.

### 4. Research Impact
- 📊 **Publication-ready**: Large-scale analysis suitable for academic papers
- 📊 **Reproducible**: Full methodology documented and automated
- 📊 **Comprehensive**: Covers multiple genres and writing contexts
- 📊 **Reliable**: Results backed by substantial data

## Dataset Characteristics

### Finance (7,740 samples)
- **Source**: Financial Q&A forums
- **Style**: Mix of casual user advice and formal financial analysis
- **Topics**: Investments, loans, mortgages, retirement planning
- **Balance**: Slightly AI-heavy (56.6% AI)

### Medicine (2,254 samples)
- **Source**: Medical consultation Q&A
- **Style**: Professional medical advice
- **Topics**: Symptoms, diagnoses, treatments, medications
- **Balance**: AI-heavy (57.7% AI)

### Open QA (3,392 samples)
- **Source**: General knowledge questions
- **Style**: Encyclopedic, factual responses
- **Topics**: Science, history, culture, technology
- **Balance**: Heavily AI-skewed (95.9% AI) ⚠️

### Reddit ELI5 (52,028 samples)
- **Source**: Reddit "Explain Like I'm 5" community
- **Style**: Casual, conversational explanations
- **Topics**: Everything from science to pop culture
- **Balance**: Human-heavy (68.1% human)
- **Note**: Largest dataset, most diverse

### Wiki CS/AI (1,622 samples)
- **Source**: Computer Science and AI topics
- **Style**: Academic/encyclopedic
- **Topics**: Algorithms, ML, programming, theory
- **Balance**: Well-balanced (51.4% AI)

## Processing Time Estimates

Based on the pipeline architecture (batch_size=64):
- **Finance (7,740)**: ~8-12 minutes
- **Medicine (2,254)**: ~3-5 minutes
- **Open QA (3,392)**: ~4-6 minutes
- **Reddit ELI5 (52,028)**: ~60-90 minutes ⏰
- **Wiki CS/AI (1,622)**: ~2-3 minutes

**Total processing time**: ~1.5-2 hours for all genres

## Output Organization

Results are saved to genre-specific folders:

```
results_HC3_finance_full/
├── human_vs_ai_augmented.csv    # All computed features
├── figures/
│   ├── figure_1_flowchart.png
│   ├── figure_2_keywords_human.png
│   └── figure_3_keywords_ai.png
└── tables/
    ├── statistical_tests.csv     # p-values, effect sizes
    ├── keywords_group_0.csv      # Human keywords
    └── keywords_group_1.csv      # AI keywords
```

## Quick Commands

### Convert all HC3 datasets (one-time setup)
```bash
python convert_all_hc3.py
```

### Process all full datasets
```bash
python main.py --dataset hc3_full
```

### Process specific genre
```bash
python main.py --dataset hc3_finance_full
python main.py --dataset hc3_medicine_full
python main.py --dataset hc3_reddit_eli5_full
```

### Compare full datasets
```bash
python main.py --compare hc3_finance_full hc3_medicine_full
```

### Validate converted files
```bash
python utils.py validate
```

## Expected Results Improvements

With full datasets, expect:

1. **Stronger Statistical Significance**
   - More features reaching p < 0.05
   - Tighter confidence intervals
   - More reliable effect sizes

2. **Clearer Patterns**
   - Keywords more representative
   - Collocation patterns more stable
   - POS ratios more consistent

3. **Better Genre Discrimination**
   - Can identify genre-specific markers
   - Cross-genre comparison more meaningful
   - Universal vs. genre-specific features

4. **Robust Conclusions**
   - Results less affected by outliers
   - More generalizable findings
   - Publication-quality evidence

## Memory Considerations

Large datasets (especially Reddit ELI5 with 52K samples) may require:
- **RAM**: 8-16 GB recommended
- **Disk Space**: ~500 MB per large result folder
- **Processing**: Multi-core CPU helpful for batch processing

If memory is limited, process smaller genres first or use:
```bash
python convert_all_hc3.py --max 5000  # Limit to 5000 per group
```

## Research Questions Enabled

With full datasets, we can now investigate:

1. **Scale Effects**: Do patterns hold at different sample sizes?
2. **Genre Specificity**: Which features are universal vs. genre-specific?
3. **Imbalance Robustness**: Do metrics work with skewed distributions?
4. **Temporal Patterns**: Are there differences across data collection periods?
5. **Cross-validation**: Can train/test splits be created for ML models?

## Next Steps After Processing

1. **Compare Genres**: Identify universal AI writing patterns
2. **Statistical Meta-Analysis**: Aggregate effect sizes across genres
3. **Feature Selection**: Identify most discriminative features
4. **Model Development**: Use features for ML classification
5. **Publication**: Results suitable for academic papers

## Citation

If using this full dataset analysis approach:

```
Zhang, Y. (2024). Nominalization in ChatGPT-generated texts compared to human-produced texts: 
Overuse, unique types, and implications for AI literacy. International Review of Applied 
Linguistics in Language Teaching (IRAL).

[Your analysis using this pipeline with full HC3 corpus]
```

## Maintenance

Keep datasets updated:
```bash
# Re-convert if source files change
python convert_all_hc3.py

# Clean old partial results
python utils.py clean --all

# Re-run analysis
python main.py --dataset hc3_full
```

# IRAL Pipeline - Quick Start Guide

## 🎯 Ultra-Simple Usage

### For HC3 Datasets (Recommended)

Convert all HC3 genres with **FULL data** (not just 100 samples):

```bash
python convert_all_hc3.py
```

Then process everything:

```bash
python main.py --dataset hc3_full
```

That's it! This gives you:
- **67,036 total samples** across 5 genres
- Finance: 7,740 samples
- Medicine: 2,254 samples  
- Open QA: 3,392 samples
- Reddit ELI5: 52,028 samples
- Wiki CS/AI: 1,622 samples

### For Custom Datasets

Just put your CSV files in `data/raw/` and run:

```bash
python main.py
```

The pipeline will:
1. ✅ Automatically discover all CSV files in `data/raw/`
2. ✅ Process each dataset through the full analysis pipeline
3. ✅ Generate organized results in `results_*` folders
4. ✅ Compare HC3 datasets automatically
5. ✅ Print a comprehensive summary

## CSV File Requirements

Your CSV files must have:
- A `text` column (or specify with `--textcol`)
- A `label` column with 0=human, 1=AI (or specify with `--labelcol`)

## Examples

### Process all datasets
```bash
python main.py
```

### Process only HC3 datasets
```bash
python main.py --dataset hc3
```

### Process only finance dataset
```bash
python main.py --dataset finance
```

### Compare two datasets (after processing)
```bash
python main.py --compare hc3_finance hc3_medicine
```

### Use custom column names
```bash
python main.py --textcol content --labelcol class
```

### Quiet mode (less output)
```bash
python main.py --quiet
```

## Output Structure

Results are automatically organized by dataset:

```
results_HC3_finance/
├── human_vs_ai_augmented.csv    # Full results with all features
├── figures/
│   ├── figure_1_flowchart.png
│   ├── figure_2_keywords_human.png
│   └── figure_3_keywords_ai.png
└── tables/
    ├── statistical_tests.csv     # All statistical comparisons
    ├── keywords_group_0.csv      # Human keywords
    └── keywords_group_1.csv      # AI keywords
```

## Advanced Options

```bash
python main.py --help
```

Shows all available options:
- `--data-dir PATH` - Custom data directory (default: data/raw)
- `--batch-size N` - Processing batch size (default: 64)
- `--seed N` - Random seed (default: 42)
- And more...

## What It Does

For each dataset, the pipeline:
1. Ingests and validates data
2. Cleans text
3. Performs POS tagging and tokenization
4. Computes linguistic features (TTR, sentence length, word length, etc.)
5. Detects nominalizations
6. Extracts collocations and keywords
7. Runs statistical tests (t-tests, Mann-Whitney U)
8. Generates IRAL-style visualizations
9. Exports comprehensive results

## Need Help?

- Check that your CSV files are in `data/raw/`
- Make sure columns are named `text` and `label` (or use `--textcol` and `--labelcol`)
- Ensure labels are 0 (human) and 1 (AI)
- Run with `--help` for all options

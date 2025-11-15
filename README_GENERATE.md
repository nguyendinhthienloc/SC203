# IRAL Python Project - Zhang (2024) Reproduction

A complete Python reproduction of Mengxuan Zhang's IRAL methodology from "More Human Than Human?"

## Features

This project implements:
- **Text ingestion** from CSV or folder of `.txt` files
- **Cleaning & normalization** with citation removal
- **Tokenization & POS-tagging** using spaCy (with NLTK fallback)
- **Noun counting** and **nominalization detection**
  - Lemma-based (spaCy)
  - Suffix-based (heuristic)
- **Lexical metrics**: TTR, mean sentence length, average word length
- **Collocations**: bigrams with PMI
- **Keywords**: log-odds with Haldane–Anscombe correction
- **Statistical tests**: Welch t-test, Mann-Whitney, Cohen's d
- **Visualizations**: boxplots, bar plots with confidence intervals
- **Export**: augmented CSV, tables, and figures

## Installation

### Local Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Google Colab

```python
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm
```

## Quick Start

### Run on CSV file

```bash
python scripts/analyze_nominalization.py \
    --input data/example.csv \
    --textcol text \
    --labelcol label \
    --outdir results/
```

### Run on folder of text files

```bash
python scripts/analyze_nominalization.py \
    --input data/raw \
    --outdir results/
```

## Project Structure

```
project_root/
├─ data/
│  ├─ raw/          # Input data (CSV or .txt files)
│  ├─ cleaned/      # Preprocessed data
│  └─ derived/      # Extracted features
├─ notebooks/       # Jupyter notebooks for exploration
├─ src/             # Core modules
│  ├─ ingest.py           # Data loading
│  ├─ clean.py            # Text cleaning
│  ├─ features.py         # Lexical metrics
│  ├─ pos_tools.py        # POS tagging
│  ├─ nominalization.py   # Nominalization detection
│  ├─ collocations.py     # Collocations & keywords
│  ├─ stats_analysis.py   # Statistical tests
│  ├─ plots.py            # Visualizations
│  └─ run_pipeline.py     # Main orchestrator
├─ scripts/
│  └─ analyze_nominalization.py  # CLI wrapper
├─ results/
│  ├─ figures/      # Generated plots
│  └─ tables/       # Generated tables
├─ tests/           # Unit tests
├─ requirements.txt
└─ README_GENERATE.md
```

## Module Details

### `src/ingest.py`
Load data from CSV or folder of text files.

### `src/clean.py`
Normalize text, remove citations and reference sections.

### `src/pos_tools.py`
Tokenization and POS tagging with spaCy (preferred) or NLTK fallback.

### `src/features.py`
Compute lexical metrics: word count, sentence count, TTR, etc.

### `src/nominalization.py`
Two detection methods:
1. **Lemma-based**: nouns whose lemmas are also verbs
2. **Suffix-based**: heuristic rules for common suffixes

### `src/collocations.py`
Extract bigrams, compute PMI, and identify keywords using log-odds ratio.

### `src/stats_analysis.py`
Statistical tests: Welch t-test, Mann-Whitney U, Cohen's d, confidence intervals.

### `src/plots.py`
Generate boxplots, bar charts, and keyword visualizations.

### `src/run_pipeline.py`
Main pipeline orchestrator that ties all modules together.

## Testing

```bash
pytest tests/
```

## Output

The pipeline generates:
- **CSV**: `human_vs_ai_augmented.csv` with all computed features
- **Figures**: Boxplots and bar charts in `results/figures/`
- **Tables**: Statistical summaries in `results/tables/`

## Citation

If using this code, please cite:

Zhang, M. (2024). More Human Than Human? Investigating ChatGPT's Linguistic Footprints on Academic Writing. *IRAL - International Review of Applied Linguistics in Language Teaching*.

## License

MIT License (or specify your preferred license)

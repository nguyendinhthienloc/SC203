# IRAL Text Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete Python reproduction of Zhang (2024)'s IRAL methodology for linguistic analysis of human vs. AI-generated academic texts, with focus on nominalization detection, lexical diversity, and statistical comparison.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Methodology & Formulas](#methodology--formulas)
- [Project Structure](#project-structure)
- [Output](#output)
- [Usage Examples](#usage-examples)
- [Citation](#citation)
- [License](#license)

## ✨ Features

This pipeline implements comprehensive linguistic analysis:

### Core Analysis
- **Text Ingestion**: CSV files or folders of `.txt` documents
- **Text Cleaning**: Citation removal, reference section filtering, normalization
- **POS Tagging**: spaCy (preferred) with NLTK fallback
- **Lexical Metrics**: Type-Token Ratio (TTR), sentence length, word length
- **Nominalization Detection**: 
  - Lemma-based (verb-derived nouns via WordNet)
  - Suffix-based heuristics (POS-filtered)
- **Collocations**: Bigram extraction with PMI scoring
- **Keywords**: Log-odds ratio with Haldane–Anscombe correction
- **Statistical Testing**: Welch's t-test, Mann-Whitney U, Cohen's d, with FDR correction
- **Visualizations**: Boxplots, bar charts with confidence intervals

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Environment (Development Install)

```bash
# Clone the repository
git clone https://github.com/nguyendinhthienloc/SC203.git
cd SC203

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Editable install (preferred for development)
pip install -e .[dev]

# Or fallback to requirements.txt
pip install -r requirements.txt

# Download language models and data
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Google Colab

```python
!git clone https://github.com/nguyendinhthienloc/SC203.git
%cd SC203
!pip install -r requirements.txt
!python -m spacy download en_core_web_sm
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 🎯 Quick Start

### Basic Usage

```bash
# Run on CSV file
python scripts/analyze_nominalization.py \
    --input data/raw/sample_data.csv \
    --textcol text \
    --labelcol label \
    --outdir results/

# Run on folder of text files
python scripts/analyze_nominalization.py \
    --input data/raw/ \
    --outdir results/

### Advanced CLI Flags

```bash
# Enable batching + multi-process (if spaCy supports it)
python scripts/analyze_nominalization.py --input my.csv --batch-size 64 --n-process 2 \
  --nominalization-mode strict --seed 42 --outdir results/

# Skip keyword extraction for faster runs
python scripts/analyze_nominalization.py --input my.csv --skip-keywords

# Force minimum frequency for keywords
python scripts/analyze_nominalization.py --input my.csv --min-freq-keywords 3

# Load overrides from YAML config (values override CLI flags)
python scripts/analyze_nominalization.py --input my.csv --config config.yml
```

Flag summary:

| Flag | Purpose |
|------|---------|
| `--batch-size` | Control spaCy pipe batch size (performance tuning) |
| `--n-process` | Parallel spaCy processes (CPU permitting) |
| `--collocation-min-count` | Minimum bigram frequency for PMI filtering |
| `--min-freq-keywords` | Override default keyword min frequency |
| `--nominalization-mode` | Detection strictness: `strict`, `balanced`, `lenient` |
| `--seed` | Deterministic mode (affects random sampling/order) |
| `--skip-keywords` | Disable keyword extraction stage entirely |
| `--verbose` / `--debug` | Logging verbosity control |
| `--config` | YAML file with flag overrides |
| `--save-intermediates` | (Reserved) Persist intermediate artifacts |

### Determinism & Reproducibility

Setting `--seed` initializes Python, NumPy, and (where applicable) other libraries for reproducible runs. All randomized ordering (e.g., document traversal) becomes stable across executions.

### Benchmarking

Run synthetic benchmarks to assess scaling:

```bash
python benchmarks/benchmark_pipeline.py --sizes 10 100 500 --repeats 2
```

Outputs CSV with mean, min, max duration (and memory deltas if `psutil` installed).
```

### Python API

```python
from src.run_pipeline import run_pipeline

# Analyze your data
results_df = run_pipeline(
    input_path="data/your_data.csv",
    textcol="text",
    labelcol="label",
    outdir="results/"
)

# Access computed features
print(results_df.columns)
# ['id', 'label', 'word_count', 'sentence_count', 'avg_sentence_len',
#  'avg_word_len', 'type_token_ratio', 'noun_count', 'verb_count',
#  'nominal_lemma_count', 'nominal_lemma_ratio', 'nominal_suffix_count', ...]
```

## 📐 Methodology & Formulas

### Lexical Metrics

```
word_count = Σ 1[token]
sentence_count = Σ 1[sentence]
avg_sentence_len = word_count / sentence_count
avg_word_len = (Σ len(token)) / word_count
type_token_ratio (TTR) = |unique_tokens| / word_count
```

### Nominalization Detection

#### Lemma-Based (Primary Method)
A noun is classified as a nominalization if:
1. Its POS tag is NOUN or PROPN (spaCy)
2. Its lemma matches a verb lemma in the document, OR
3. WordNet lists a derivationally-related verb form

```
nominal_ratio = count(verb-derived nouns) / count(all nouns)
```

#### Suffix-Based (Heuristic)
Detects common nominalization suffixes in noun-tagged tokens:
- `tion`, `sion`, `ment`, `ence`, `ance`
- `ity`, `ness`, `ship`, `age`, `al`, `ure`, `ing`

```
suffix_count = Σ 1[token ends with suffix ∧ POS ∈ {NOUN, PROPN}]
```

### Collocations

**Pointwise Mutual Information (PMI)**:
```
PMI(x,y) = log₂(P(x,y) / (P(x) × P(y)))

where:
  P(x,y) = count(bigram) / total_bigrams
  P(x) = count(word_x) / total_words
```

### Keywords

**Log-Odds Ratio with Haldane–Anscombe Correction**:
```
log_odds = log₂((f_A + c) / (N_A + c×V)) / ((f_B + c) / (N_B + c×V))

where:
  f_x = frequency in corpus x
  N_x = total tokens in corpus x
  V = vocabulary size
  c = correction factor (0.5)
```

### Statistical Tests

**Welch's t-test** (unequal variances):
```
t = (x̄_A - x̄_B) / √(s²_A/n_A + s²_B/n_B)
```

**Cohen's d** (effect size):
```
d = (x̄_A - x̄_B) / s_pooled

where:
  s_pooled = √(((n_A-1)×s²_A + (n_B-1)×s²_B) / (n_A + n_B - 2))
```

**FDR Correction** (Benjamini-Hochberg):
- Applied to adjust p-values for multiple comparisons
- Exported as `p_value_adj` and `mw_p_value_adj` in results

## 📁 Project Structure

```
SC203/
├── data/
│   ├── raw/              # Input CSV or .txt files
│   ├── cleaned/          # Preprocessed texts
│   └── derived/          # Intermediate features
├── documents/            # Project documentation
├── notebooks/            # Jupyter notebooks for exploration
├── results/
│   ├── figures/          # Boxplots, bar charts, keyword plots
│   └── tables/           # Statistical tests, keywords CSV
├── scripts/
│   └── analyze_nominalization.py  # Command-line interface
├── src/
│   ├── __init__.py
│   ├── ingest.py         # Data loading (CSV/folder)
│   ├── clean.py          # Text preprocessing
│   ├── pos_tools.py      # Tokenization & POS tagging
│   ├── features.py       # Lexical metrics computation
│   ├── nominalization.py # Nominalization detection
│   ├── collocations.py   # Bigrams, PMI, keywords
│   ├── stats_analysis.py # Statistical tests & FDR
│   ├── plots.py          # Visualization generation
│   └── run_pipeline.py   # Main orchestrator
├── tests/
│   ├── test_nominalization.py
│   └── test_pipeline_smoke.py
├── .gitignore
├── README.md
├── requirements.txt
└── environment.yml
```

## 📊 Output

The pipeline generates:

### 1. Augmented CSV (`results/human_vs_ai_augmented.csv`)
Contains all computed features per document:
- Basic metrics: word_count, sentence_count, TTR
- POS counts and ratios
- Nominalization counts and ratios
- Top collocations with PMI scores

### 2. Statistical Tables (`results/tables/`)
- `statistical_tests.csv`: Welch t-test, Mann-Whitney U, Cohen's d, confidence intervals, adjusted p-values
- `keywords_group_0.csv`: Keywords distinctive to group 0
- `keywords_group_1.csv`: Keywords distinctive to group 1

### 3. Visualizations (`results/figures/`)
- Boxplots for each metric by group
- Bar charts with 95% confidence intervals
- Keyword plots (horizontal bar charts with log-odds scores)

## 💡 Usage Examples

### Example 1: Analyze Academic Papers

```python
from src.run_pipeline import run_pipeline
import pandas as pd

# Prepare your data
df = pd.DataFrame({
    'text': ['Human-written paper text...', 'AI-generated paper text...'],
    'label': [0, 1]  # 0=human, 1=AI
})
df.to_csv('my_data.csv', index=False)

# Run analysis
results = run_pipeline('my_data.csv', outdir='my_results/')
```

### Example 2: Compare Two Corpora

```python
# Place human texts in data/human/*.txt
# Place AI texts in data/ai/*.txt
# The pipeline infers labels from filenames containing 'human' or 'ai'

results = run_pipeline('data/', outdir='corpus_comparison/')
```

### Example 3: Custom Feature Extraction

```python
from src.pos_tools import tokenize_and_pos
from src.features import compute_basic_metrics
from src.nominalization import analyze_nominalization

text = "The implementation of the system requires development."

# Tokenize and tag
pos_result = tokenize_and_pos(text)

# Compute metrics
metrics = compute_basic_metrics(pos_result['words'], pos_result['sentences'])
print(f"TTR: {metrics['type_token_ratio']}")

# Detect nominalizations
nominals = analyze_nominalization(
    doc=pos_result['doc'],
    tokens=pos_result['words'],
    pos_tokens=pos_result.get('pos_tokens')
)
print(f"Nominalizations: {nominals['lemma_based']['nominal_from_verb']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_nominalization.py -v

# Run with coverage
pytest --cov=src tests/
```

### Formatting & Lint

```bash
black .
isort .
flake8 src scripts tests
```

## 📚 Citation

If you use this code in your research, please cite:

**Original IRAL Study:**
```bibtex
@article{zhang2024human,
  title={More Human Than Human? Investigating ChatGPT's Linguistic Footprints on Academic Writing},
  author={Zhang, Mengxuan},
  journal={IRAL - International Review of Applied Linguistics in Language Teaching},
  year={2024},
  publisher={De Gruyter}
}
```

**This Implementation:**
```bibtex
@software{sc203_iral,
  author={Nguyen Dinh Thien Loc},
  title={SC203: IRAL Text Analysis Pipeline},
  year={2024},
  url={https://github.com/nguyendinhthienloc/SC203}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Issues

If you encounter any problems or have suggestions, please [open an issue](https://github.com/nguyendinhthienloc/SC203/issues).

## 📧 Contact

Nguyen Dinh Thien Loc - [GitHub](https://github.com/nguyendinhthienloc)

---

**Note**: This implementation extends Zhang (2024) with additional features including WordNet-based nominalization detection, FDR correction for multiple comparisons, and comprehensive visualization capabilities.

## 🔄 Changelog

See `CHANGELOG.md` for a full list of versions and enhancements.

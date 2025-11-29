# IRAL Text Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python implementation of Zhang (2024)'s IRAL methodology for linguistic analysis of human vs. AI-generated texts, focusing on nominalization detection, lexical diversity, and statistical comparison.

> 📖 **[See PROJECT_GUIDE.md for comprehensive documentation](PROJECT_GUIDE.md)** - Detailed explanations of all files, data science concepts, NLP metrics, and future plans.

## ✨ Features

- **Nominalization Detection**: Lemma-based + suffix heuristics (key AI writing marker)
- **Lexical Analysis**: TTR, sentence length, word length, POS ratios
- **Statistical Testing**: Welch's t-test, Mann-Whitney U, Cohen's d, FDR correction
- **Collocations & Keywords**: PMI scoring, log-odds ratio
- **Publication Figures**: IRAL-style visualizations ready for papers

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/nguyendinhthienloc/SC203.git
cd SC203

# Setup environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -e .[dev]

# Download models
python -m spacy download en_core_web_sm
```

### Run Analysis

**One-button run:**
```bash
python run.py  # Processes data/raw/sample_data.csv
```

**Command-line interface:**
```bash
python scripts/analyze_nominalization.py \
    --input data/raw/your_data.csv \
    --textcol text \
    --labelcol label \
    --outdir results/
```

**Python API:**
```python
from src.run_pipeline import run_pipeline

results = run_pipeline(
    input_path="data/your_data.csv",
    textcol="text",
    labelcol="label",
    outdir="results/"
)
```

### Output Structure

```
results/
├── human_vs_ai_augmented.csv       # All computed features
├── figures/
│   ├── figure_1_flowchart.png      # Pipeline visualization
│   ├── figure_2_keywords_human.png # Human-distinctive words
│   └── figure_3_keywords_ai.png    # AI-distinctive words
└── tables/
    ├── statistical_tests.csv        # p-values, effect sizes
    ├── keywords_group_0.csv
    └── keywords_group_1.csv
```

## 📊 Metric Comparison: Zhang (2024) vs. This Reproduction

| Source         | Metric Families / Types | Description |
|----------------|------------------------|-------------|
| **Zhang (2024)** | 3                    | Word Frequency, Collocations (Count, Lambda, Z-score), Keyword Analysis (Log-Odds Ratio) |
| **This Project** | 11                   | Word Count, Sentence Count, TTR, Mean Sentence Length, Average Word Length, POS Distributions, Nominalization Count (lemma-based), Nominalization Count (suffix-based), Bigram PMI, Keyword Log-Odds Ratio, Effect Size & Significance Tests |

- Zhang’s metrics are grouped into 3 families (word frequency, collocations, keywords).
- This reproduction expands to 11 distinct metric types, including morphosyntactic, lexical, collocation, keyword, and statistical comparison metrics.
- New metrics/features: POS distributions, nominalization (suffix-based, strict/balanced/lenient), effect size/statistical tests, and more detailed lexical metrics.

## 📐 Key Metrics

**Nominalization Ratio** (Primary metric)
```
nominal_ratio = verb-derived_nouns / total_nouns
```
- Zhang (2024) finding: AI texts show **higher nominalization** (42-48% vs. 32-38%)
- Indicates more formal, abstracted writing style

**Lexical Features**
- Type-Token Ratio (TTR): Vocabulary diversity
- Average sentence length: Syntactic complexity
- POS ratios: Noun vs. verb usage patterns

**Statistical Tests**
- Welch's t-test (unequal variances)
- Mann-Whitney U (non-parametric)
- Cohen's d (effect size)
- FDR-BH correction (multiple comparisons)

**Collocation & Keywords**
- PMI: Word association strength
- Log-odds: Distinctive vocabulary per group

> **For detailed formulas and methodology, see [PROJECT_GUIDE.md](PROJECT_GUIDE.md)**

## 📁 Project Structure

```
SC203/
├── run.py                    # 👈 One-button launcher
├── src/
│   ├── run_pipeline.py       # Main orchestrator
│   ├── ingest.py             # Data loading
│   ├── clean.py              # Text preprocessing
│   ├── pos_tools.py          # POS tagging
│   ├── features.py           # Lexical metrics
│   ├── nominalization.py     # Nominalization detection
│   ├── collocations.py       # Collocations & keywords
│   ├── stats_analysis.py     # Statistical tests
│   └── plots*.py             # Visualizations
├── scripts/
│   └── analyze_nominalization.py  # CLI interface
├── tests/                    # Comprehensive test suite
├── data/
│   ├── raw/                  # 👈 Put your data here
│   └── HC3/                  # Sample dataset
├── results/                  # 👈 Output goes here
└── PROJECT_GUIDE.md          # 👈 Detailed documentation
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📚 Citation

**Original Study:**
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

## 📖 Documentation

- **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)** - Complete technical guide
  - Detailed file documentation
  - Data science concepts explained
  - NLP metrics in research context
  - Statistical methods deep dive
  - Future development roadmap

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 🤝 Contributing

Contributions welcome! Please submit Pull Requests or open Issues.

## 📧 Contact

Nguyen Dinh Thien Loc - [GitHub](https://github.com/nguyendinhthienloc)

---

**Note**: This implementation extends Zhang (2024) with FDR correction, enhanced visualizations, and comprehensive testing for production-ready research applications.

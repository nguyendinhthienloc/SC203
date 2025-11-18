# Quick Start Guide 🚀

## One-Button Run

### Option 1: Double-click `RUN.bat` (Windows)
The easiest way! Just double-click the `RUN.bat` file in your file explorer.

### Option 2: Right-click `RUN.ps1` → Run with PowerShell
Alternative for PowerShell users.

### Option 3: Command Line
```powershell
python run.py
```

## What It Does

The launcher will:
1. ✅ Load your data from `data/raw/sample_data.csv`
2. ✅ Clean and process all texts
3. ✅ Extract nominalization features
4. ✅ Compute collocations and keywords
5. ✅ Run statistical tests
6. ✅ Generate all figures
7. ✅ Save results to `results/`

## Output Structure

```
results/
├── figures/
│   ├── figure_1_flowchart.png
│   ├── figure_2_keywords_human.png
│   └── figure_3_keywords_ai.png
├── tables/
│   ├── statistical_tests.csv
│   ├── keywords_group_0.csv
│   └── keywords_group_1.csv
└── human_vs_ai_augmented.csv
```

## First-Time Setup

If you haven't installed dependencies yet:

```powershell
# Create virtual environment (one-time)
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install dependencies (one-time)
pip install -e .[dev]

# Download spaCy model (one-time)
python -m spacy download en_core_web_sm
```

## Change Input Data

Edit `run.py` line 22 to point to your data file:

```python
input_path = "data/raw/sample_data.csv"  # Change this
```

Or use the full CLI:

```powershell
python scripts/analyze_nominalization.py --input YOUR_FILE.csv --outdir results/
```

## Troubleshooting

**"Module not found" error?**
→ Run: `pip install -e .[dev]`

**"Virtual environment not found"?**
→ Run: `python -m venv venv`

**Need different settings?**
→ Edit parameters in `run.py` or use the CLI with flags (see README.md)

## Quick Stats

- ⚡ Processes ~4 documents in ~1 second
- 📊 Generates 12 statistical tests
- 🎨 Creates 3 publication-ready figures
- 📈 Extracts 15-20 keywords per group

#!/usr/bin/env python
"""
One-button launcher for IRAL pipeline.
Run this to process your data and generate all results.

Usage:
    python run.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.run_pipeline import run_pipeline


def main():
    """Run the full pipeline on HC3 data."""
    print("=" * 80)
    print("IRAL Text Analysis Pipeline - One-Button Run")
    print("=" * 80)
    
    # Configuration
    input_path = "data/raw/sample_data.csv"
    textcol = "text"
    labelcol = "label"
    outdir = "results"
    
    print(f"\n📂 Input: {input_path}")
    print(f"📊 Output: {outdir}/")
    print(f"🔧 Mode: balanced nominalization, seed=42, batch_size=64")
    print(f"⚠️  GECS features: DISABLED (use enable_gecs=True to enable)")
    print(f"    Note: GECS requires OpenAI API and takes longer to run")
    print("\nStarting analysis...\n")
    
    try:
        results_df = run_pipeline(
            input_path=input_path,
            textcol=textcol,
            labelcol=labelcol,
            outdir=outdir,
            nominalization_mode="balanced",
            collocation_min_count=5,
            skip_keywords=False,
            min_freq_keywords=None,
            batch_size=64,
            n_process=1,
            seed=42,
            verbose=True,
            debug=False
        )
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Analysis completed")
        print("=" * 80)
        print(f"\n📊 Processed {len(results_df)} documents")
        print(f"\n📁 Results saved to: {outdir}/")
        print(f"   - Augmented data: {outdir}/human_vs_ai_augmented.csv")
        print(f"   - Statistical tests: {outdir}/tables/statistical_tests.csv")
        print(f"   - Keywords: {outdir}/tables/keywords_group_*.csv")
        print(f"   - Figures: {outdir}/figures/*.png")
        print("\n✨ You can now explore the results!")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Input file not found")
        print(f"   {e}")
        print(f"\n💡 Make sure your data file exists at: {input_path}")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

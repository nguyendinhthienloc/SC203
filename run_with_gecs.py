#!/usr/bin/env python
"""
Run IRAL pipeline with GECS (Grammar Error Correction Score) features enabled.

This script runs the full IRAL analysis with additional GECS features:
- Corrects grammar using GPT-4o
- Calculates Rouge-2 similarity scores
- Adds GEC scores to statistical analysis

⚠️  WARNING: This requires OpenAI API key and will incur API costs!
    Estimated cost: ~$0.01-0.05 per 100 documents (using gpt-4o-mini)

Usage:
    python run_with_gecs.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.run_pipeline import run_pipeline


def main():
    """Run the full pipeline with GECS features enabled."""
    print("=" * 80)
    print("IRAL Text Analysis Pipeline - WITH GECS FEATURES")
    print("=" * 80)
    
    # Configuration
    input_path = "data/HC3/hc3_sample.json"  # Can also use CSV
    textcol = "text"
    labelcol = "label"
    outdir = "results_with_gecs"
    
    print(f"\n📂 Input: {input_path}")
    print(f"📊 Output: {outdir}/")
    print(f"🔧 Mode: balanced nominalization, seed=42, batch_size=64")
    print(f"✅ GECS features: ENABLED (using gpt-4o-mini)")
    print(f"\n⚠️  WARNING: This will use OpenAI API and incur costs!")
    print(f"   Estimated: ~$0.01-0.05 per 100 documents")
    
    # Ask for confirmation
    response = input("\nProceed? [y/N]: ").strip().lower()
    if response != 'y':
        print("Aborted.")
        return 0
    
    print("\nStarting analysis with GECS...\n")
    print("📝 Note: GECS computation may take several minutes...")
    print("   Progress will be shown as documents are processed.\n")
    
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
            debug=False,
            enable_gecs=True,  # 🔥 Enable GECS features
            gecs_model="gpt-4o-mini"  # Use cheaper model
        )
        
        print("\n" + "=" * 80)
        print("✅ SUCCESS! Analysis with GECS completed")
        print("=" * 80)
        print(f"\n📊 Processed {len(results_df)} documents")
        print(f"\n📁 Results saved to: {outdir}/")
        print(f"   - Augmented data: {outdir}/human_vs_ai_augmented.csv")
        print(f"   - Statistical tests: {outdir}/tables/statistical_tests.csv")
        print(f"   - Keywords: {outdir}/tables/keywords_group_*.csv")
        print(f"   - Figures: {outdir}/figures/*.png")
        
        # Show GECS-specific info
        if 'gec_rouge2_score' in results_df.columns:
            gecs_count = results_df['gec_rouge2_score'].notna().sum()
            print(f"\n🎯 GECS Features:")
            print(f"   - Computed for {gecs_count}/{len(results_df)} documents")
            if gecs_count > 0:
                mean_score = results_df['gec_rouge2_score'].mean()
                print(f"   - Mean Rouge-2 score: {mean_score:.4f}")
                
                # By label
                if 'label' in results_df.columns:
                    for label in sorted(results_df['label'].unique()):
                        label_scores = results_df[results_df['label'] == label]['gec_rouge2_score']
                        label_name = "Human" if label == 0 else "AI" if label == 1 else f"Group {label}"
                        print(f"   - {label_name}: {label_scores.mean():.4f} ± {label_scores.std():.4f}")
        
        print("\n✨ You can now explore the results!")
        print(f"   Check {outdir}/tables/statistical_tests.csv for GECS statistics")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Input file not found")
        print(f"   {e}")
        print(f"\n💡 Make sure your data file exists at: {input_path}")
        return 1
    
    except ImportError as e:
        print(f"\n❌ ERROR: Missing required package")
        print(f"   {e}")
        print(f"\n💡 Install with: pip install openai rouge")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

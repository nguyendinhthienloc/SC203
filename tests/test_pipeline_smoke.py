"""
Smoke tests for the complete pipeline.
"""

import sys
import os
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.run_pipeline import run_pipeline


def test_pipeline_smoke():
    """
    Smoke test: run pipeline end-to-end with minimal data.
    """
    print("Running pipeline smoke test...\n")
    
    # Create test data
    test_data = [
        {
            "text": "Human writing often involves development and implementation of complex ideas. "
                   "The analysis requires careful consideration and reference to existing literature.",
            "label": 0
        },
        {
            "text": "AI system implementation text with automation and generation capabilities. "
                   "The creation of content through artificial intelligence systems.",
            "label": 1
        },
        {
            "text": "Another human text discussing the relationship between theory and practice. "
                   "The application of knowledge requires understanding and interpretation.",
            "label": 0
        },
        {
            "text": "More AI generated content with systematic organization and presentation. "
                   "The information processing demonstrates technical capability.",
            "label": 1
        }
    ]
    
    df = pd.DataFrame(test_data)
    
    # Save test CSV
    test_csv = "test_input.csv"
    df.to_csv(test_csv, index=False)
    
    print(f"Created test data: {test_csv}")
    
    # Run pipeline
    try:
        results_df = run_pipeline(
            input_path=test_csv,
            textcol="text",
            labelcol="label",
            outdir="results_test"
        )
        
        # Verify outputs
        assert os.path.exists("results_test/human_vs_ai_augmented.csv"), \
            "Output CSV not found"
        
        assert os.path.exists("results_test/figures"), \
            "Figures directory not created"
        
        assert os.path.exists("results_test/tables"), \
            "Tables directory not created"
        
        # Verify dataframe
        assert len(results_df) == len(df), \
            f"Expected {len(df)} rows, got {len(results_df)}"
        
        # Check key columns exist
        expected_columns = [
            'word_count', 'sentence_count', 'avg_sentence_len',
            'type_token_ratio', 'noun_count', 'verb_count',
            'nominal_lemma_count', 'nominal_suffix_count'
        ]
        
        for col in expected_columns:
            assert col in results_df.columns, f"Column '{col}' not found in results"
        
        print("\n" + "=" * 50)
        print("✓ Pipeline smoke test passed!")
        print("=" * 50)
        print(f"\nResults summary:")
        print(f"  - Processed {len(results_df)} documents")
        print(f"  - Generated {len(results_df.columns)} features")
        print(f"  - Output saved to: results_test/")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
            print(f"\nCleaned up: {test_csv}")
        
        if os.path.exists("results_test"):
            # Uncomment to remove test results
            # shutil.rmtree("results_test")
            # print(f"Cleaned up: results_test/")
            print(f"Test results kept in: results_test/")


def test_pipeline_minimal():
    """
    Minimal test with just 2 documents.
    """
    print("\nRunning minimal pipeline test...\n")
    
    df = pd.DataFrame([
        {"text": "Short human text with development.", "label": 0},
        {"text": "Short AI text with implementation.", "label": 1}
    ])
    
    test_csv = "test_minimal.csv"
    df.to_csv(test_csv, index=False)
    
    try:
        results_df = run_pipeline(test_csv, "text", "label", "results_minimal")
        
        assert len(results_df) == 2
        assert 'word_count' in results_df.columns
        
        print("✓ Minimal test passed")
        return True
        
    except Exception as e:
        print(f"✗ Minimal test failed: {e}")
        return False
        
    finally:
        if os.path.exists(test_csv):
            os.remove(test_csv)


if __name__ == "__main__":
    print("=" * 50)
    print("Running Pipeline Tests")
    print("=" * 50 + "\n")
    
    test1 = test_pipeline_smoke()
    test2 = test_pipeline_minimal()
    
    if test1 and test2:
        print("\n" + "=" * 50)
        print("All pipeline tests passed!")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed")
        print("=" * 50)
        sys.exit(1)

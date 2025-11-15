"""
Command-line interface for IRAL nominalization analysis.

Usage:
    python scripts/analyze_nominalization.py --input data/example.csv --textcol text --labelcol label --outdir results/
    python scripts/analyze_nominalization.py --input data/raw --outdir results/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.run_pipeline import run_pipeline


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="IRAL Text Analysis Pipeline - Zhang (2024) Reproduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze CSV file
  python scripts/analyze_nominalization.py --input data/example.csv --textcol text --labelcol label --outdir results/
  
  # Analyze folder of text files
  python scripts/analyze_nominalization.py --input data/raw --outdir results/
  
  # Use default columns
  python scripts/analyze_nominalization.py --input data/example.csv --outdir results/
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file or folder of .txt files'
    )
    
    parser.add_argument(
        '--textcol',
        type=str,
        default='text',
        help='Column name for text content (CSV only, default: text)'
    )
    
    parser.add_argument(
        '--labelcol',
        type=str,
        default='label',
        help='Column name for labels (CSV only, default: label)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Run pipeline
    try:
        results_df = run_pipeline(
            input_path=args.input,
            textcol=args.textcol,
            labelcol=args.labelcol,
            outdir=args.outdir
        )
        
        print(f"\n✓ Analysis completed successfully!")
        print(f"✓ Processed {len(results_df)} documents")
        
    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

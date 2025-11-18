"""
Command-line interface for IRAL nominalization analysis.

Usage:
    python scripts/analyze_nominalization.py --input data/example.csv --textcol text --labelcol label --outdir results/
    python scripts/analyze_nominalization.py --input data/raw --outdir results/
"""

import argparse
import sys
from pathlib import Path
import yaml

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

    parser.add_argument('--batch-size', type=int, default=32, help='spaCy pipe batch size (default 32)')
    parser.add_argument('--n-process', type=int, default=1, help='spaCy n_process (default 1)')
    parser.add_argument('--collocation-min-count', type=int, default=5, help='Min bigram count for PMI (default 5)')
    parser.add_argument('--min-freq-keywords', type=int, default=None, help='Override min freq for keywords')
    parser.add_argument('--nominalization-mode', type=str, default='balanced', choices=['strict','balanced','lenient'],
                        help='Nominalization detection mode')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for deterministic mode')
    parser.add_argument('--skip-keywords', action='store_true', help='Skip keyword extraction stage')
    parser.add_argument('--verbose', action='store_true', help='Enable INFO logging')
    parser.add_argument('--debug', action='store_true', help='Enable DEBUG logging')
    parser.add_argument('--config', type=str, default=None, help='Optional YAML config file to override flags')
    parser.add_argument('--save-intermediates', action='store_true', help='(Reserved) Save intermediate artifacts')
    
    args = parser.parse_args()

    # Load config overrides if provided
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: Config file '{cfg_path}' not found")
            sys.exit(1)
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            for k, v in cfg.items():
                if hasattr(args, k):
                    setattr(args, k, v)
        except Exception as e:
            print(f"Error parsing config file: {e}")
            sys.exit(1)
    
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
            outdir=args.outdir,
            nominalization_mode=args.nominalization_mode,
            collocation_min_count=args.collocation_min_count,
            skip_keywords=args.skip_keywords,
            min_freq_keywords=args.min_freq_keywords,
            batch_size=args.batch_size,
            n_process=args.n_process,
            seed=args.seed,
            verbose=args.verbose or args.debug,
            debug=args.debug
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

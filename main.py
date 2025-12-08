#!/usr/bin/env python
"""
Main entry point for IRAL Text Analysis Pipeline.

This script automatically:
1. Discovers all CSV files in data/raw/
2. Processes each dataset through the full pipeline
3. Generates comparison reports between datasets
4. Exports all results to organized folders

Usage:
    python main.py                    # Process all datasets
    python main.py --dataset finance  # Process specific dataset
    python main.py --compare          # Only run comparison (skip processing)
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.run_pipeline import run_pipeline


class PipelineManager:
    """Manages pipeline execution for multiple datasets."""
    
    def __init__(self, data_dir: str = "data/raw", results_base: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_base = Path(results_base)
        self.processed_datasets: List[Dict] = []
        
    def discover_datasets(self, pattern: Optional[str] = None) -> List[Path]:
        """
        Discover CSV files in data directory.
        
        Parameters
        ----------
        pattern : str, optional
            Filter datasets by name pattern (e.g., 'hc3' or 'finance')
            
        Returns
        -------
        List[Path]
            List of CSV file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if pattern:
            csv_files = [f for f in csv_files if pattern.lower() in f.stem.lower()]
        
        return sorted(csv_files)
    
    def get_output_dir(self, dataset_path: Path) -> Path:
        """
        Generate output directory name from dataset filename.
        
        Examples:
            hc3_finance.csv -> results_HC3_finance
            hc3_medicine.csv -> results_HC3_medicine
            sample_data.csv -> results_sample
        """
        stem = dataset_path.stem
        
        # Handle HC3 datasets specially
        if stem.startswith('hc3_'):
            genre = stem.replace('hc3_', '')
            return self.results_base.parent / f"results_HC3_{genre}"
        else:
            return self.results_base.parent / f"results_{stem}"
    
    def process_dataset(
        self,
        dataset_path: Path,
        textcol: str = "text",
        labelcol: str = "label",
        nominalization_mode: str = "balanced",
        batch_size: int = 64,
        seed: int = 42,
        verbose: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Process a single dataset through the pipeline.
        
        Parameters
        ----------
        dataset_path : Path
            Path to CSV file
        textcol : str
            Column name for text data
        labelcol : str
            Column name for labels (0=human, 1=AI)
        nominalization_mode : str
            Mode for nominalization analysis
        batch_size : int
            Batch size for processing
        seed : int
            Random seed for reproducibility
        verbose : bool
            Print progress messages
            
        Returns
        -------
        pd.DataFrame or None
            Augmented results dataframe, or None if processing failed
        """
        outdir = self.get_output_dir(dataset_path)
        
        print("\n" + "=" * 80)
        print(f"Processing: {dataset_path.name}")
        print("=" * 80)
        print(f"📂 Input:  {dataset_path}")
        print(f"📊 Output: {outdir}/")
        print(f"🔧 Config: mode={nominalization_mode}, seed={seed}, batch={batch_size}")
        print()
        
        try:
            results_df = run_pipeline(
                input_path=str(dataset_path),
                textcol=textcol,
                labelcol=labelcol,
                outdir=str(outdir),
                nominalization_mode=nominalization_mode,
                collocation_min_count=5,
                skip_keywords=False,
                min_freq_keywords=None,
                batch_size=batch_size,
                n_process=1,
                seed=seed,
                verbose=verbose,
                debug=False
            )
            
            # Track successful processing
            self.processed_datasets.append({
                'name': dataset_path.stem,
                'path': dataset_path,
                'output_dir': outdir,
                'n_documents': len(results_df),
                'status': 'success'
            })
            
            print("\n" + "=" * 80)
            print(f"✅ SUCCESS: {dataset_path.name}")
            print("=" * 80)
            print(f"📊 Processed {len(results_df)} documents")
            print(f"📁 Results: {outdir}/")
            
            return results_df
            
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"❌ FAILED: {dataset_path.name}")
            print("=" * 80)
            print(f"Error: {e}")
            
            self.processed_datasets.append({
                'name': dataset_path.stem,
                'path': dataset_path,
                'output_dir': None,
                'n_documents': 0,
                'status': 'failed',
                'error': str(e)
            })
            
            if verbose:
                import traceback
                traceback.print_exc()
            
            return None
    
    def process_all(self, pattern: Optional[str] = None, **kwargs) -> int:
        """
        Process all discovered datasets.
        
        Parameters
        ----------
        pattern : str, optional
            Filter datasets by name pattern
        **kwargs
            Additional arguments passed to process_dataset
            
        Returns
        -------
        int
            Number of successfully processed datasets
        """
        datasets = self.discover_datasets(pattern)
        
        if not datasets:
            print(f"\n⚠️  No CSV files found in {self.data_dir}/")
            if pattern:
                print(f"   (with pattern: '{pattern}')")
            return 0
        
        print("\n" + "=" * 80)
        print(f"IRAL Pipeline - Batch Processing")
        print("=" * 80)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Found {len(datasets)} dataset(s):")
        for ds in datasets:
            print(f"   • {ds.name}")
        print()
        
        success_count = 0
        
        for dataset in datasets:
            result = self.process_dataset(dataset, **kwargs)
            if result is not None:
                success_count += 1
        
        return success_count
    
    def print_summary(self):
        """Print summary of all processed datasets."""
        if not self.processed_datasets:
            print("\nNo datasets processed yet.")
            return
        
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        
        success = [d for d in self.processed_datasets if d['status'] == 'success']
        failed = [d for d in self.processed_datasets if d['status'] == 'failed']
        
        print(f"\n✅ Successful: {len(success)}/{len(self.processed_datasets)}")
        for ds in success:
            print(f"   • {ds['name']:<30} ({ds['n_documents']} docs) → {ds['output_dir']}")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)}/{len(self.processed_datasets)}")
            for ds in failed:
                print(f"   • {ds['name']:<30} Error: {ds.get('error', 'Unknown')}")
        
        print("\n" + "=" * 80)
    
    def compare_results(self, dataset1: str, dataset2: str):
        """
        Compare results between two processed datasets.
        
        Parameters
        ----------
        dataset1 : str
            Name of first dataset (e.g., 'hc3_finance')
        dataset2 : str
            Name of second dataset (e.g., 'hc3_medicine')
        """
        # Find output directories
        ds1_info = next((d for d in self.processed_datasets if d['name'] == dataset1), None)
        ds2_info = next((d for d in self.processed_datasets if d['name'] == dataset2), None)
        
        if not ds1_info or not ds2_info:
            print(f"\n⚠️  Cannot compare: one or both datasets not found")
            print(f"   Available: {[d['name'] for d in self.processed_datasets]}")
            return
        
        dir1 = ds1_info['output_dir']
        dir2 = ds2_info['output_dir']
        
        stats1_path = dir1 / "tables" / "statistical_tests.csv"
        stats2_path = dir2 / "tables" / "statistical_tests.csv"
        
        if not stats1_path.exists() or not stats2_path.exists():
            print(f"\n⚠️  Cannot compare: statistical test results not found")
            return
        
        print("\n" + "=" * 80)
        print(f"COMPARISON: {dataset1} vs {dataset2}")
        print("=" * 80)
        
        stats1 = pd.read_csv(stats1_path)
        stats2 = pd.read_csv(stats2_path)
        
        sig1 = stats1[stats1['p_value_adj'] < 0.05]
        sig2 = stats2[stats2['p_value_adj'] < 0.05]
        
        print(f"\n{dataset1}: {len(sig1)}/{len(stats1)} significant features")
        print(f"{dataset2}: {len(sig2)}/{len(stats2)} significant features")
        
        print(f"\nEffect sizes (Cohen's d):")
        print(f"{'Metric':<25} {dataset1:>15} {dataset2:>15} {'Stronger in':>15}")
        print("-" * 75)
        
        for _, row1 in stats1.iterrows():
            metric = row1['metric']
            row2 = stats2[stats2['metric'] == metric].iloc[0]
            
            d1 = abs(row1['cohen_d'])
            d2 = abs(row2['cohen_d'])
            stronger = dataset1 if d1 > d2 else dataset2
            
            print(f"{metric:<25} {row1['cohen_d']:>15.3f} {row2['cohen_d']:>15.3f} {stronger:>15}")
        
        print("\n" + "=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="IRAL Text Analysis Pipeline - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Process all datasets in data/raw/
  python main.py --dataset hc3            # Process only HC3 datasets
  python main.py --dataset finance        # Process only finance dataset
  python main.py --compare finance medicine  # Compare two datasets
  python main.py --textcol content --labelcol class  # Custom column names
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Process only datasets matching this pattern (e.g., "hc3", "finance")'
    )
    
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('DATASET1', 'DATASET2'),
        help='Compare results between two datasets (must be processed first)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing CSV files (default: data/raw)'
    )
    
    parser.add_argument(
        '--textcol',
        type=str,
        default='text',
        help='Column name for text data (default: text)'
    )
    
    parser.add_argument(
        '--labelcol',
        type=str,
        default='label',
        help='Column name for labels (default: label)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for processing (default: 64)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = PipelineManager(data_dir=args.data_dir)
    
    try:
        if args.compare:
            # Comparison mode
            dataset1, dataset2 = args.compare
            
            # First, make sure datasets are in processed list
            # (Re-discover to populate the list if needed)
            for dataset_name in [dataset1, dataset2]:
                datasets = manager.discover_datasets(dataset_name)
                if datasets:
                    output_dir = manager.get_output_dir(datasets[0])
                    if output_dir.exists():
                        manager.processed_datasets.append({
                            'name': datasets[0].stem,
                            'path': datasets[0],
                            'output_dir': output_dir,
                            'status': 'success'
                        })
            
            manager.compare_results(dataset1, dataset2)
        else:
            # Processing mode
            success_count = manager.process_all(
                pattern=args.dataset,
                textcol=args.textcol,
                labelcol=args.labelcol,
                batch_size=args.batch_size,
                seed=args.seed,
                verbose=not args.quiet
            )
            
            manager.print_summary()
            
            # Auto-compare HC3 datasets if multiple were processed
            hc3_datasets = [d for d in manager.processed_datasets 
                           if d['status'] == 'success' and 'hc3_' in d['name']]
            
            if len(hc3_datasets) >= 2:
                print("\n💡 Multiple HC3 datasets detected. Running comparison...")
                for i in range(len(hc3_datasets) - 1):
                    manager.compare_results(
                        hc3_datasets[i]['name'],
                        hc3_datasets[i + 1]['name']
                    )
            
            print(f"\n✨ All done! Processed {success_count} dataset(s)")
            print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            return 0 if success_count > 0 else 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

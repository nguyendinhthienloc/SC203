#!/usr/bin/env python
"""
Utility script for managing IRAL pipeline files.

Usage:
    python utils.py list              # List all result folders
    python utils.py clean             # Clean old temporary files
    python utils.py clean --all       # Remove ALL result folders (careful!)
    python utils.py validate          # Validate all CSV files in data/raw/
"""

import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd


def list_results(verbose=False):
    """List all result folders."""
    root = Path(".")
    result_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("results")])
    
    if not result_dirs:
        print("No result folders found.")
        return
    
    print(f"\n📊 Found {len(result_dirs)} result folder(s):\n")
    
    for result_dir in result_dirs:
        size = sum(f.stat().st_size for f in result_dir.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # Count files
        csv_files = len(list(result_dir.rglob('*.csv')))
        png_files = len(list(result_dir.rglob('*.png')))
        
        print(f"📁 {result_dir.name}/")
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Files: {csv_files} CSV, {png_files} PNG")
        
        if verbose:
            # Show modification time
            mtime = result_dir.stat().st_mtime
            mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"   Modified: {mtime_str}")
        
        print()


def clean_temp_files():
    """Remove temporary files and old scripts."""
    temp_patterns = [
        "test_input.csv",
        "test_output",
        "run_finance.py",
        "evaluate_results.py",
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        ".coverage"
    ]
    
    print("\n🧹 Cleaning temporary files...\n")
    
    removed_count = 0
    for pattern in temp_patterns:
        for path in Path(".").rglob(pattern):
            try:
                if path.is_file():
                    path.unlink()
                    print(f"   ✓ Removed: {path}")
                    removed_count += 1
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"   ✓ Removed directory: {path}")
                    removed_count += 1
            except Exception as e:
                print(f"   ✗ Could not remove {path}: {e}")
    
    print(f"\n✨ Cleaned {removed_count} item(s)")


def clean_all_results(force=False):
    """Remove ALL result folders."""
    root = Path(".")
    result_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("results")]
    
    if not result_dirs:
        print("No result folders to remove.")
        return
    
    if not force:
        print(f"\n⚠️  WARNING: This will DELETE {len(result_dirs)} result folder(s):")
        for d in result_dirs:
            print(f"   • {d.name}/")
        
        response = input("\nAre you sure? Type 'yes' to confirm: ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    print(f"\n🗑️  Removing {len(result_dirs)} result folder(s)...\n")
    
    for result_dir in result_dirs:
        try:
            shutil.rmtree(result_dir)
            print(f"   ✓ Removed: {result_dir.name}/")
        except Exception as e:
            print(f"   ✗ Could not remove {result_dir}: {e}")
    
    print("\n✨ Done")


def validate_csv_files():
    """Validate all CSV files in data/raw/."""
    data_dir = Path("data/raw")
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}/")
        return
    
    print(f"\n🔍 Validating {len(csv_files)} CSV file(s)...\n")
    
    valid_count = 0
    
    for csv_file in csv_files:
        print(f"📄 {csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {list(df.columns)}")
            
            # Check required columns
            has_text = 'text' in df.columns
            has_label = 'label' in df.columns
            
            if has_text:
                print(f"   ✓ 'text' column found")
            else:
                print(f"   ⚠️  'text' column missing (use --textcol to specify)")
            
            if has_label:
                print(f"   ✓ 'label' column found")
                labels = df['label'].unique()
                print(f"   Labels: {sorted(labels)}")
                
                if set(labels) == {0, 1}:
                    print(f"   ✓ Valid labels (0=human, 1=AI)")
                    label_dist = df['label'].value_counts().to_dict()
                    print(f"   Distribution: {label_dist}")
                else:
                    print(f"   ⚠️  Expected labels: 0 and 1")
            else:
                print(f"   ⚠️  'label' column missing (use --labelcol to specify)")
            
            # Check for empty texts
            if has_text:
                empty_count = df['text'].isna().sum()
                if empty_count > 0:
                    print(f"   ⚠️  {empty_count} empty text entries")
                else:
                    print(f"   ✓ No empty texts")
            
            if has_text and has_label:
                valid_count += 1
                print(f"   ✅ Valid dataset")
            else:
                print(f"   ❌ Missing required columns")
            
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
        
        print()
    
    print(f"✨ {valid_count}/{len(csv_files)} valid dataset(s)")


def show_disk_usage():
    """Show disk usage of project directories."""
    directories = ["data", "results_HC3_finance", "results_HC3_medicine", "results_sample", "results_sample_data"]
    
    print("\n💾 Disk Usage:\n")
    
    total_size = 0
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
        
        size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        total_size += size
        
        print(f"📁 {dir_name:<25} {size_mb:>10.2f} MB")
    
    print(f"\n{'Total':<25} {total_size / (1024 * 1024):>10.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="IRAL Pipeline Utility Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'command',
        choices=['list', 'clean', 'validate', 'disk'],
        help='Command to execute: list (show results), clean (remove temp files), validate (check CSVs), disk (show usage)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='For clean command: remove ALL result folders'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show more details'
    )
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_results(verbose=args.verbose)
    elif args.command == 'clean':
        if args.all:
            clean_all_results(force=args.force)
        else:
            clean_temp_files()
    elif args.command == 'validate':
        validate_csv_files()
    elif args.command == 'disk':
        show_disk_usage()


if __name__ == "__main__":
    main()

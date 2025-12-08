#!/usr/bin/env python
"""
Convert ALL HC3 JSONL datasets to CSV format for comprehensive analysis.

This script processes all HC3 genres with their full data (not just 100 samples)
and saves them to data/raw/ for pipeline processing.

Usage:
    python convert_all_hc3.py                # Convert all genres with full data
    python convert_all_hc3.py --max 500      # Limit to 500 samples per group
    python convert_all_hc3.py --genre finance medicine  # Specific genres only
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def load_jsonl(file_path):
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def convert_hc3_to_csv(input_path, output_path, max_samples=None, min_words=50):
    """
    Convert HC3 JSONL to CSV format.
    
    Parameters
    ----------
    input_path : Path
        Path to HC3 JSONL file
    output_path : Path
        Path to output CSV file
    max_samples : int, optional
        Maximum number of samples per group (human/AI), None = all data
    min_words : int, default=50
        Minimum word count for text to be included
        
    Returns
    -------
    pd.DataFrame
        The converted dataframe
    """
    print(f"\n{'='*80}")
    print(f"Converting: {input_path.name}")
    print(f"{'='*80}")
    print(f"Loading data...")
    
    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} question-answer pairs")
    
    # Extract texts
    texts = []
    labels = []
    ids = []
    
    id_counter = 1
    human_count = 0
    ai_count = 0
    
    human_skipped = 0
    ai_skipped = 0
    
    for item in data:
        # Extract human answers
        if 'human_answers' in item and item['human_answers']:
            for answer in item['human_answers']:
                word_count = len(answer.split())
                if word_count >= min_words:
                    if max_samples is None or human_count < max_samples:
                        texts.append(answer)
                        labels.append(0)  # 0 = human
                        ids.append(f"human_{id_counter}")
                        id_counter += 1
                        human_count += 1
                else:
                    human_skipped += 1
        
        # Extract ChatGPT answers
        if 'chatgpt_answers' in item and item['chatgpt_answers']:
            for answer in item['chatgpt_answers']:
                word_count = len(answer.split())
                if word_count >= min_words:
                    if max_samples is None or ai_count < max_samples:
                        texts.append(answer)
                        labels.append(1)  # 1 = AI
                        ids.append(f"ai_{id_counter}")
                        id_counter += 1
                        ai_count += 1
                else:
                    ai_skipped += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'text': texts,
        'label': labels
    })
    
    # Calculate statistics
    human_texts = df[df['label'] == 0]['text']
    ai_texts = df[df['label'] == 1]['text']
    
    human_avg_words = human_texts.apply(lambda x: len(x.split())).mean()
    ai_avg_words = ai_texts.apply(lambda x: len(x.split())).mean()
    
    print(f"\n📊 Extraction Results:")
    print(f"  Human (label=0): {human_count:,} samples (avg {human_avg_words:.1f} words)")
    print(f"  AI (label=1):    {ai_count:,} samples (avg {ai_avg_words:.1f} words)")
    print(f"  Total:           {len(df):,} samples")
    
    if human_skipped or ai_skipped:
        print(f"\n  Skipped (< {min_words} words):")
        print(f"    Human: {human_skipped:,}")
        print(f"    AI:    {ai_skipped:,}")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Saved to: {output_path}")
    
    # Show sample
    print(f"\n📄 Sample texts:")
    print(f"  Human (first 150 chars): {human_texts.iloc[0][:150]}...")
    print(f"  AI (first 150 chars):    {ai_texts.iloc[0][:150]}...")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert all HC3 JSONL datasets to CSV format with FULL data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--genre',
        nargs='+',
        choices=['finance', 'medicine', 'open_qa', 'reddit_eli5', 'wiki_csai'],
        help='Specific genres to convert (default: all)'
    )
    
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Maximum samples per group (human/AI). Default: None (use all data)'
    )
    
    parser.add_argument(
        '--min-words',
        type=int,
        default=50,
        help='Minimum word count for texts (default: 50)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for CSV files (default: data/raw)'
    )
    
    args = parser.parse_args()
    
    # Define available genres
    hc3_dir = Path('data/HC3')
    
    available_genres = {
        'finance': hc3_dir / 'finance.jsonl',
        'medicine': hc3_dir / 'medicine.jsonl',
        'open_qa': hc3_dir / 'open_qa.jsonl',
        'reddit_eli5': hc3_dir / 'reddit_eli5.jsonl',
        'wiki_csai': hc3_dir / 'wiki_csai.jsonl'
    }
    
    # Determine which genres to process
    if args.genre:
        genres_to_process = {k: v for k, v in available_genres.items() if k in args.genre}
    else:
        genres_to_process = available_genres
    
    if not genres_to_process:
        print("No genres selected for conversion.")
        return 1
    
    print("\n" + "="*80)
    print("HC3 Dataset Conversion - FULL DATA")
    print("="*80)
    print(f"📁 Input:  {hc3_dir}")
    print(f"📊 Output: {args.output_dir}")
    print(f"🔧 Config: max_samples={'all' if args.max is None else args.max}, min_words={args.min_words}")
    print(f"\n📋 Processing {len(genres_to_process)} genre(s):")
    for genre in genres_to_process.keys():
        print(f"   • {genre}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for genre, input_path in genres_to_process.items():
        if not input_path.exists():
            print(f"\n⚠️  File not found: {input_path}")
            continue
        
        output_path = output_dir / f"hc3_{genre}_full.csv"
        
        try:
            df = convert_hc3_to_csv(
                input_path=input_path,
                output_path=output_path,
                max_samples=args.max,
                min_words=args.min_words
            )
            
            results.append({
                'genre': genre,
                'samples': len(df),
                'output': output_path,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n❌ Error converting {genre}: {e}")
            results.append({
                'genre': genre,
                'samples': 0,
                'output': None,
                'status': 'failed'
            })
    
    # Print summary
    print("\n" + "="*80)
    print("CONVERSION SUMMARY")
    print("="*80)
    
    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\n✅ Successfully converted: {len(success)}/{len(results)}")
    for r in success:
        print(f"   • {r['genre']:<15} {r['samples']:>6,} samples → {r['output'].name}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}")
        for r in failed:
            print(f"   • {r['genre']}")
    
    total_samples = sum(r['samples'] for r in success)
    print(f"\n📊 Total samples: {total_samples:,}")
    
    print("\n" + "="*80)
    print("✨ Conversion complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Run: python main.py --dataset hc3_full")
    print(f"  2. This will process all {len(success)} genre(s) automatically")
    print(f"  3. Results will be saved to results_HC3_<genre>_full/ folders")
    
    return 0 if len(success) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

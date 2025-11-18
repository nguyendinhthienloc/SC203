"""
Convert HC3 JSONL dataset to CSV format for IRAL analysis pipeline.

The HC3 dataset contains human and ChatGPT answers to medical questions.
This script extracts the answers and creates a labeled dataset.

Usage:
    python scripts/convert_hc3_to_csv.py --input data/HC3/medicine.jsonl --output data/raw/hc3_medicine.csv --max-samples 100
"""

import json
import pandas as pd
import argparse
from pathlib import Path


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
    input_path : str
        Path to HC3 JSONL file
    output_path : str
        Path to output CSV file
    max_samples : int, optional
        Maximum number of samples per group (human/AI)
    min_words : int, default=50
        Minimum word count for text to be included
    """
    print(f"Loading HC3 data from: {input_path}")
    data = load_jsonl(input_path)
    
    print(f"Loaded {len(data)} question-answer pairs")
    
    # Extract texts
    texts = []
    labels = []
    ids = []
    
    id_counter = 1
    human_count = 0
    ai_count = 0
    
    for item in data:
        # Extract human answers
        if 'human_answers' in item and item['human_answers']:
            for answer in item['human_answers']:
                # Filter by minimum word count
                word_count = len(answer.split())
                if word_count >= min_words:
                    if max_samples is None or human_count < max_samples:
                        texts.append(answer)
                        labels.append(0)  # 0 = human
                        ids.append(f"human_{id_counter}")
                        id_counter += 1
                        human_count += 1
        
        # Extract ChatGPT answers
        if 'chatgpt_answers' in item and item['chatgpt_answers']:
            for answer in item['chatgpt_answers']:
                # Filter by minimum word count
                word_count = len(answer.split())
                if word_count >= min_words:
                    if max_samples is None or ai_count < max_samples:
                        texts.append(answer)
                        labels.append(1)  # 1 = AI
                        ids.append(f"ai_{id_counter}")
                        id_counter += 1
                        ai_count += 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'text': texts,
        'label': labels
    })
    
    print(f"\nExtracted texts:")
    print(f"  Human (label=0): {human_count} samples")
    print(f"  AI (label=1): {ai_count} samples")
    print(f"  Total: {len(df)} samples")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✓ Saved to: {output_path}")
    
    # Show sample
    print(f"\nSample texts:")
    print(f"\nHuman example (first 200 chars):")
    print(df[df['label'] == 0]['text'].iloc[0][:200] + "...")
    print(f"\nAI example (first 200 chars):")
    print(df[df['label'] == 1]['text'].iloc[0][:200] + "...")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert HC3 JSONL dataset to CSV format for IRAL pipeline"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to HC3 JSONL file (e.g., data/HC3/medicine.jsonl)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file (e.g., data/raw/hc3_medicine.csv)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples per group (default: all)'
    )
    
    parser.add_argument(
        '--min-words',
        type=int,
        default=50,
        help='Minimum word count for texts (default: 50)'
    )
    
    args = parser.parse_args()
    
    convert_hc3_to_csv(
        args.input,
        args.output,
        max_samples=args.max_samples,
        min_words=args.min_words
    )


if __name__ == "__main__":
    main()

"""
Convert HC3 CSV data to JSON format for GEC scoring script.

Usage:
    python scripts/convert_csv_to_json_gecs.py --input results_HC3/human_vs_ai_augmented.csv --output data/HC3/hc3_data.json
"""

import pandas as pd
import json
import argparse
from pathlib import Path


def convert_csv_to_json(input_path, output_path, max_samples=None):
    """
    Convert CSV to JSON format for GECS script.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file
    output_path : str
        Path to output JSON file
    max_samples : int, optional
        Maximum number of samples per group (human/AI)
    """
    print(f"Loading CSV data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} samples")
    
    # Create JSON format
    data = []
    
    # Filter by label
    human_df = df[df['label'] == 0]
    ai_df = df[df['label'] == 1]
    
    # Apply max_samples if specified
    if max_samples:
        human_df = human_df.head(max_samples)
        ai_df = ai_df.head(max_samples)
    
    # Process human texts
    for idx, row in human_df.iterrows():
        # Get text from the 'text' column if it exists, otherwise use id
        if 'text' in df.columns:
            text = str(row['text'])
        else:
            # If text column doesn't exist, we need to find it in the original data
            text = f"Sample text for {row['id']}"
        
        data.append({
            "id": row['id'],
            "text": text,
            "label": "human"
        })
    
    # Process AI texts
    for idx, row in ai_df.iterrows():
        if 'text' in df.columns:
            text = str(row['text'])
        else:
            text = f"Sample text for {row['id']}"
        
        data.append({
            "id": row['id'],
            "text": text,
            "label": "llm"
        })
    
    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nConverted data:")
    print(f"  Human samples: {len(human_df)}")
    print(f"  AI samples: {len(ai_df)}")
    print(f"  Total: {len(data)}")
    print(f"\n✓ Saved to: {output_path}")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Convert CSV to JSON format for GECS script"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSON file'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples per group (default: all)'
    )
    
    args = parser.parse_args()
    
    convert_csv_to_json(
        args.input,
        args.output,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

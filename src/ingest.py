"""
Data ingestion module for loading texts from CSV or folder of .txt files.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Tuple


def ingest(input_path, textcol="text", labelcol="label"):
    """
    Load data from CSV file or folder of text files.
    
    Parameters
    ----------
    input_path : str
        Path to CSV file or folder containing .txt files
    textcol : str, default="text"
        Column name for text content (CSV only)
    labelcol : str, default="label"
        Column name for labels (CSV only)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id, text, label
    """
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Load from CSV or JSONL
        if input_path.suffix == '.jsonl':
            df = pd.read_json(input_path, lines=True)
        else:
            df = pd.read_csv(input_path)
        
        # Ensure required columns exist
        if textcol not in df.columns:
            raise ValueError(f"Column '{textcol}' not found in file")
        
        # Create standardized dataframe
        result = pd.DataFrame({
            'id': df.index if 'id' not in df.columns else df['id'],
            'text': df[textcol],
            'label': df[labelcol] if labelcol in df.columns else 0
        })
        
        return result
    
    elif input_path.is_dir():
        # Load from folder of text files
        texts = []
        labels = []
        ids = []
        
        txt_files = list(input_path.glob("*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {input_path}")
        
        for i, txt_file in enumerate(sorted(txt_files)):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            texts.append(text)
            ids.append(txt_file.stem)
            
            # Try to infer label from filename (e.g., human_001.txt, ai_002.txt)
            filename_lower = txt_file.stem.lower()
            if 'human' in filename_lower:
                labels.append(0)
            elif 'ai' in filename_lower or 'chatgpt' in filename_lower:
                labels.append(1)
            else:
                labels.append(0)  # default
        
        result = pd.DataFrame({
            'id': ids,
            'text': texts,
            'label': labels
        })
        
        return result
    
    else:
        raise ValueError(f"Input path {input_path} is neither a file nor a directory")


def validate_inputs(df: pd.DataFrame, textcol: str = "text", labelcol: str = "label") -> pd.DataFrame:
    """Validate and normalize input DataFrame.

    Checks:
    - Required text column exists (raise if missing)
    - Label column: if missing, create with default 0 and warn
    - Drop rows with empty/whitespace-only text (warn count)
    - Auto-generate 'id' column if missing
    - Normalize labels: map {'human','ai'} or {'Human','AI'} to {0,1}
    - Warn if only one unique label present

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    textcol : str
        Name of text column.
    labelcol : str
        Name of label column.

    Returns
    -------
    pd.DataFrame
        Validated and possibly modified dataframe.
    """
    # Text column presence
    if textcol not in df.columns:
        raise ValueError(f"Required text column '{textcol}' not found.")

    # Ensure id column
    if 'id' not in df.columns:
        df = df.copy()
        df['id'] = range(1, len(df) + 1)
        print(f"WARNING: 'id' column missing. Generated sequential ids.")

    # Ensure label column
    if labelcol not in df.columns:
        df = df.copy()
        df[labelcol] = 0
        print(f"WARNING: Label column '{labelcol}' missing. Assigned default label 0 to all rows.")

    # Drop empty texts
    mask_empty = df[textcol].isna() | (df[textcol].astype(str).str.strip() == "")
    empty_count = mask_empty.sum()
    if empty_count:
        df = df.loc[~mask_empty].copy()
        print(f"WARNING: Dropped {empty_count} empty/blank texts.")

    # Normalize labels if textual
    unique_labels = set(df[labelcol].astype(str).str.lower())
    mapping = None
    if unique_labels.issubset({"human", "ai"}):
        mapping = {"human": 0, "ai": 1}
    elif unique_labels.issubset({"0", "1"}):
        mapping = {"0": 0, "1": 1}

    if mapping:
        df[labelcol] = df[labelcol].astype(str).str.lower().map(mapping).astype(int)
    else:
        # Attempt numeric coercion, fallback keep as-is
        try:
            df[labelcol] = pd.to_numeric(df[labelcol], errors='ignore')
        except Exception:
            pass

    # Warn if only one label
    nunique = df[labelcol].nunique()
    if nunique < 2:
        print(f"WARNING: Only one unique label present (n={nunique}). Group comparisons will be skipped.")

    return df

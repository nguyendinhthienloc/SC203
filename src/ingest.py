"""
Data ingestion module for loading texts from CSV or folder of .txt files.
"""

import os
import pandas as pd
from pathlib import Path


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
        # Load from CSV
        df = pd.read_csv(input_path)
        
        # Ensure required columns exist
        if textcol not in df.columns:
            raise ValueError(f"Column '{textcol}' not found in CSV")
        
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

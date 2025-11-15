"""
Text cleaning and normalization module.
"""

import re


def normalize_text(text):
    """
    Normalize text while preserving punctuation for POS-tagging.
    
    Parameters
    ----------
    text : str
        Raw text to normalize
    
    Returns
    -------
    str
        Normalized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace while preserving single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_reference_section(text):
    """
    Remove reference sections from academic texts.
    
    Parameters
    ----------
    text : str
        Text potentially containing a references section
    
    Returns
    -------
    str
        Text with references section removed
    """
    if not isinstance(text, str):
        return ""
    
    # Common reference section headers (case-insensitive)
    ref_patterns = [
        r'\n\s*references\s*\n',
        r'\n\s*bibliography\s*\n',
        r'\n\s*works cited\s*\n',
        r'\n\s*literature cited\s*\n'
    ]
    
    text_lower = text.lower()
    min_position = len(text)
    
    for pattern in ref_patterns:
        match = re.search(pattern, text_lower)
        if match:
            min_position = min(min_position, match.start())
    
    # Remove everything after the reference section
    if min_position < len(text):
        text = text[:min_position]
    
    return text


def remove_citations(text):
    """
    Remove in-text citations from academic texts.
    
    Patterns removed:
    - (Author, Year)
    - (Author et al., Year)
    - [1], [2-5]
    - (Smith et al. 2020)
    
    Parameters
    ----------
    text : str
        Text containing citations
    
    Returns
    -------
    str
        Text with citations removed
    """
    if not isinstance(text, str):
        return ""
    
    # Remove parenthetical citations: (Author, Year), (Author et al., Year)
    text = re.sub(r'\([A-Z][a-zA-Z\s,&]+,?\s*\d{4}[a-z]?\)', '', text)
    text = re.sub(r'\([A-Z][a-zA-Z\s,&]+et al\.,?\s*\d{4}[a-z]?\)', '', text)
    
    # Remove numbered citations: [1], [2-5], [1,2,3]
    text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


def clean_text(text):
    """
    Complete text cleaning pipeline.
    
    Parameters
    ----------
    text : str
        Raw text
    
    Returns
    -------
    str
        Cleaned and normalized text
    """
    text = remove_reference_section(text)
    text = remove_citations(text)
    text = normalize_text(text)
    return text

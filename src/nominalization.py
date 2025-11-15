"""
Nominalization detection module.

Two methods:
1. spaCy lemma-based: nouns whose lemmas are also verbs
2. Suffix-based heuristic: common nominalization suffixes
"""

from typing import Dict, List, Optional


# Common nominalization suffixes
NOMINALIZATION_SUFFIXES = [
    'tion', 'sion', 'ment', 'ence', 'ance', 
    'ity', 'ness', 'ship', 'age', 'al', 'ure', 'ing'
]


def detect_nominals_spacy(doc):
    """
    Detect nominalizations using spaCy lemma-based method.
    
    A noun is considered a nominalization if:
    - It is tagged as NOUN or PROPN
    - Its lemma also appears as a VERB lemma in the document
    
    Parameters
    ----------
    doc : spacy.Doc
        spaCy Doc object
    
    Returns
    -------
    dict
        Dictionary containing:
        - all_noun_count: total number of nouns
        - nominal_from_verb: number of nominalizations (verb-derived)
        - nominal_ratio: proportion of nominalizations
        - examples: list of (token, lemma) tuples for nominalizations
    """
    if doc is None:
        return {
            'all_noun_count': 0,
            'nominal_from_verb': 0,
            'nominal_ratio': 0.0,
            'examples': []
        }
    
    # Collect all verb lemmas
    verb_lemmas = set()
    for token in doc:
        if token.pos_ == 'VERB' and not token.is_punct and not token.is_space:
            verb_lemmas.add(token.lemma_.lower())
    
    # Find nouns whose lemmas are also verbs
    all_nouns = []
    nominalizations = []
    
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN') and not token.is_punct and not token.is_space:
            all_nouns.append(token)
            
            if token.lemma_.lower() in verb_lemmas:
                nominalizations.append((token.text, token.lemma_))
    
    all_noun_count = len(all_nouns)
    nominal_count = len(nominalizations)
    
    return {
        'all_noun_count': all_noun_count,
        'nominal_from_verb': nominal_count,
        'nominal_ratio': round(nominal_count / all_noun_count, 4) if all_noun_count > 0 else 0.0,
        'examples': nominalizations[:20]  # Limit examples
    }


def detect_nominals_suffix(tokens):
    """
    Detect nominalizations using suffix-based heuristic.
    
    Parameters
    ----------
    tokens : list of str
        List of word tokens
    
    Returns
    -------
    dict
        Dictionary containing:
        - all_noun_count: set to len(tokens) as approximation
        - nominal_from_verb: total count of tokens matching suffixes
        - suffix_counts: dict of suffix -> count
        - examples: list of tokens matching suffixes
    """
    if not tokens:
        return {
            'all_noun_count': 0,
            'nominal_from_verb': 0,
            'suffix_counts': {},
            'examples': []
        }
    
    suffix_counts = {suffix: 0 for suffix in NOMINALIZATION_SUFFIXES}
    nominal_tokens = []
    
    for token in tokens:
        token_lower = token.lower()
        
        # Check each suffix
        for suffix in NOMINALIZATION_SUFFIXES:
            if token_lower.endswith(suffix) and len(token) > len(suffix):
                suffix_counts[suffix] += 1
                nominal_tokens.append(token)
                break  # Count each token only once
    
    total_nominals = len(nominal_tokens)
    
    return {
        'all_noun_count': len(tokens),  # Approximation
        'nominal_from_verb': total_nominals,
        'suffix_counts': suffix_counts,
        'examples': list(set(nominal_tokens))[:20]  # Unique examples, limited
    }


def analyze_nominalization(doc=None, tokens=None):
    """
    Comprehensive nominalization analysis using available methods.
    
    Parameters
    ----------
    doc : spacy.Doc, optional
        spaCy Doc object for lemma-based detection
    tokens : list of str, optional
        List of tokens for suffix-based detection
    
    Returns
    -------
    dict
        Combined results from both methods
    """
    results = {
        'lemma_based': None,
        'suffix_based': None
    }
    
    if doc is not None:
        results['lemma_based'] = detect_nominals_spacy(doc)
    
    if tokens is not None:
        results['suffix_based'] = detect_nominals_suffix(tokens)
    
    return results

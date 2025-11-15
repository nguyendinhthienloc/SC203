"""Lexical feature computations used across the IRAL replication.

Formulas implemented:
- **word_count** = Σ 1[token]
- **sentence_count** = Σ 1[sentence]
- **avg_sentence_len** = word_count / sentence_count
- **avg_word_len** = (Σ len(token)) / word_count
- **type_token_ratio** = |unique(tokens)| / word_count
"""

from typing import List


def compute_basic_metrics(tokens, sentences):
    """
    Compute basic lexical metrics.
    
    Parameters
    ----------
    tokens : list of str
        List of word tokens
    sentences : list of list of str
        List of sentences (each sentence is a list of tokens)
    
    Returns
    -------
    dict
        Dictionary containing:
        - word_count: total number of words
        - sentence_count: total number of sentences
        - avg_sentence_len: average sentence length in words
        - avg_word_len: average word length in characters
        - type_token_ratio: TTR (unique words / total words)
    """
    if not tokens:
        return {
            'word_count': 0,
            'sentence_count': 0,
            'avg_sentence_len': 0.0,
            'avg_word_len': 0.0,
            'type_token_ratio': 0.0
        }
    
    word_count = len(tokens)
    sentence_count = len([s for s in sentences if s])  # Non-empty sentences
    
    # Average sentence length
    if sentence_count > 0:
        avg_sentence_len = word_count / sentence_count
    else:
        avg_sentence_len = 0.0
    
    # Average word length
    total_char_count = sum(len(word) for word in tokens)
    avg_word_len = total_char_count / word_count if word_count > 0 else 0.0
    
    # Type-Token Ratio (TTR)
    unique_words = set(word.lower() for word in tokens)
    type_token_ratio = len(unique_words) / word_count if word_count > 0 else 0.0
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_sentence_len': round(avg_sentence_len, 2),
        'avg_word_len': round(avg_word_len, 2),
        'type_token_ratio': round(type_token_ratio, 4)
    }


def compute_pos_features(pos_counts, total_words):
    """
    Compute POS-based features.
    
    Parameters
    ----------
    pos_counts : dict
        Dictionary of POS tag counts
    total_words : int
        Total number of words
    
    Returns
    -------
    dict
        Dictionary containing counts plus ratios computed as count / total_words.
    """
    if total_words == 0:
        return {
            'noun_count': 0,
            'verb_count': 0,
            'adj_count': 0,
            'adv_count': 0,
            'noun_ratio': 0.0,
            'verb_ratio': 0.0
        }
    
    noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
    verb_count = pos_counts.get('VERB', 0)
    adj_count = pos_counts.get('ADJ', 0)
    adv_count = pos_counts.get('ADV', 0)
    
    return {
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adj_count': adj_count,
        'adv_count': adv_count,
        'noun_ratio': round(noun_count / total_words, 4),
        'verb_ratio': round(verb_count / total_words, 4)
    }

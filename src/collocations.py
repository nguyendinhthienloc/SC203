"""Collocations and keyword extraction module.

Key formulas:
- **PMI(x,y)** = log2( P(x,y) / (P(x) * P(y)) ) with probabilities estimated from corpus counts.
- **Log-odds with Haldane-Anscombe** adds 0.5 pseudo-count to each cell before computing odds ratios.
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


def bigram_counts(tokens):
    """
    Extract bigram counts from token list.
    
    Parameters
    ----------
    tokens : list of str
        List of word tokens
    
    Returns
    -------
    Counter
        Counter of (word1, word2) bigrams
    """
    bigrams = []
    for i in range(len(tokens) - 1):
        bigram = (tokens[i].lower(), tokens[i + 1].lower())
        bigrams.append(bigram)
    
    return Counter(bigrams)


def compute_pmi(bigram_counts, unigram_counts, total_bigrams, min_count=5):
    """
    Compute Pointwise Mutual Information (PMI) for bigrams.
    
    PMI(x,y) = log2(P(x,y) / (P(x) * P(y)))
    
    Parameters
    ----------
    bigram_counts : Counter
        Counter of (word1, word2) bigrams
    unigram_counts : Counter
        Counter of individual words
    total_bigrams : int
        Total number of bigrams
    min_count : int, default=5
        Minimum frequency threshold for bigrams
    
    Returns
    -------
    list of tuple
        List of ((word1, word2), pmi_score) sorted by PMI descending
    """
    if total_bigrams == 0:
        return []
    
    total_words = sum(unigram_counts.values())
    
    pmi_scores = []
    
    for bigram, count in bigram_counts.items():
        if count < min_count:
            continue
        
        word1, word2 = bigram
        
        # P(x,y)
        p_xy = count / total_bigrams
        
        # P(x) and P(y)
        p_x = unigram_counts[word1] / total_words
        p_y = unigram_counts[word2] / total_words
        
        # PMI calculation
        if p_x > 0 and p_y > 0:
            pmi = math.log2(p_xy / (p_x * p_y))
            pmi_scores.append((bigram, pmi))
    
    # Sort by PMI descending
    pmi_scores.sort(key=lambda x: x[1], reverse=True)
    
    return pmi_scores


def extract_collocations(tokens, top_n=50):
    """
    Extract top collocations from token list.
    
    Parameters
    ----------
    tokens : list of str
        List of word tokens
    top_n : int, default=50
        Number of top collocations to return
    
    Returns
    -------
    dict
        Dictionary containing:
        - bigrams: list of ((word1, word2), pmi_score)
        - top_collocations: list of top_n collocations
    """
    if len(tokens) < 2:
        return {
            'bigrams': [],
            'top_collocations': []
        }
    
    # Count bigrams and unigrams
    bigrams = bigram_counts(tokens)
    unigrams = Counter(t.lower() for t in tokens)
    
    total_bigrams = len(tokens) - 1
    
    # Compute PMI
    pmi_scores = compute_pmi(bigrams, unigrams, total_bigrams)
    
    return {
        'bigrams': pmi_scores,
        'top_collocations': pmi_scores[:top_n]
    }


def log_odds_ratio(counts_A, counts_B, correction=0.5):
    """
    Compute log-odds ratio with Haldane-Anscombe correction.
    
    Log-odds ratio measures keyness of words between two corpora.
    Formula: log2( ( (f_A + c) / (N_A + c*V) ) / ( (f_B + c) / (N_B + c*V) ) )
    where c is the correction term, N_x is total tokens in corpus x, V is vocab size.
    
    Parameters
    ----------
    counts_A : Counter
        Word counts in corpus A
    counts_B : Counter
        Word counts in corpus B
    correction : float, default=0.5
        Haldane-Anscombe correction factor (usually 0.5)
    
    Returns
    -------
    dict
        Dictionary with:
        - keywords_A: list of (word, log_odds) for corpus A
        - keywords_B: list of (word, log_odds) for corpus B
    """
    total_A = sum(counts_A.values())
    total_B = sum(counts_B.values())
    
    if total_A == 0 or total_B == 0:
        return {
            'keywords_A': [],
            'keywords_B': []
        }
    
    # Get all unique words
    all_words = set(counts_A.keys()) | set(counts_B.keys())
    
    log_odds_scores = {}
    
    for word in all_words:
        # Apply Haldane-Anscombe correction
        freq_A = counts_A.get(word, 0) + correction
        freq_B = counts_B.get(word, 0) + correction
        
        # Normalize by corpus size
        prop_A = freq_A / (total_A + correction * len(all_words))
        prop_B = freq_B / (total_B + correction * len(all_words))
        
        # Log-odds ratio
        log_odds = math.log2(prop_A / prop_B)
        
        log_odds_scores[word] = log_odds
    
    # Sort by log-odds
    sorted_scores = sorted(log_odds_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Split into keywords for each corpus
    keywords_A = [(word, score) for word, score in sorted_scores if score > 0]
    keywords_B = [(word, -score) for word, score in reversed(sorted_scores) if score < 0]
    
    return {
        'keywords_A': keywords_A[:100],  # Top 100 keywords
        'keywords_B': keywords_B[:100]
    }


def extract_keywords(tokens_A, tokens_B, min_freq=5):
    """
    Extract keywords distinguishing two text groups.
    
    Parameters
    ----------
    tokens_A : list of str
        Tokens from corpus A (e.g., human-written)
    tokens_B : list of str
        Tokens from corpus B (e.g., AI-generated)
    min_freq : int, default=5
        Minimum frequency threshold
    
    Returns
    -------
    dict
        Keywords for each corpus with log-odds scores
    """
    # Count words in each corpus
    counts_A = Counter(t.lower() for t in tokens_A)
    counts_B = Counter(t.lower() for t in tokens_B)
    
    # Filter by minimum frequency
    counts_A = Counter({w: c for w, c in counts_A.items() if c >= min_freq})
    counts_B = Counter({w: c for w, c in counts_B.items() if c >= min_freq})
    
    # Compute log-odds
    keywords = log_odds_ratio(counts_A, counts_B)
    
    return keywords

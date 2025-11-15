"""
POS tagging module with spaCy (preferred) and NLTK fallback.
"""

import re
from typing import Dict, List, Tuple, Optional


def tokenize_and_pos(text):
    """
    Tokenize and POS-tag text using spaCy (preferred) or NLTK (fallback).
    
    Parameters
    ----------
    text : str
        Text to tokenize and tag
    
    Returns
    -------
    dict
        Dictionary containing:
        - words: list of str
        - sentences: list of list of str
        - pos_counts: dict of POS tag counts
        - doc: spaCy Doc object (or None if using NLTK)
    """
    # Try spaCy first
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            return _tokenize_spacy(text, nlp)
        except OSError:
            # Model not downloaded, try NLTK
            pass
    except ImportError:
        pass
    
    # Fallback to NLTK
    try:
        import nltk
        return _tokenize_nltk(text)
    except ImportError:
        raise ImportError("Neither spaCy nor NLTK is available. Please install at least one.")


def _tokenize_spacy(text, nlp):
    """
    Tokenize using spaCy.
    
    Parameters
    ----------
    text : str
        Text to tokenize
    nlp : spacy.Language
        spaCy language model
    
    Returns
    -------
    dict
        Tokenization results with spaCy Doc
    """
    doc = nlp(text)
    
    # Extract words (excluding punctuation and whitespace)
    words = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    # Extract sentences
    sentences = [[token.text for token in sent if not token.is_punct and not token.is_space] 
                 for sent in doc.sents]
    
    # Count POS tags
    pos_counts = {}
    for token in doc:
        if not token.is_punct and not token.is_space:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    return {
        'words': words,
        'sentences': sentences,
        'pos_counts': pos_counts,
        'doc': doc
    }


def _tokenize_nltk(text):
    """
    Tokenize using NLTK.
    
    Parameters
    ----------
    text : str
        Text to tokenize
    
    Returns
    -------
    dict
        Tokenization results without spaCy Doc
    """
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    
    # Download required resources if not available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    # Tokenize sentences
    sent_texts = sent_tokenize(text)
    
    # Tokenize words in each sentence
    sentences = []
    all_words = []
    
    for sent in sent_texts:
        words = word_tokenize(sent)
        # Filter out pure punctuation
        words = [w for w in words if re.search(r'\w', w)]
        if words:
            sentences.append(words)
            all_words.extend(words)
    
    # POS tagging
    tagged = nltk.pos_tag(all_words)
    
    # Count POS tags (convert Penn Treebank to simplified tags)
    pos_counts = {}
    for word, tag in tagged:
        # Map Penn Treebank tags to Universal POS tags
        universal_tag = _penn_to_universal(tag)
        pos_counts[universal_tag] = pos_counts.get(universal_tag, 0) + 1
    
    return {
        'words': all_words,
        'sentences': sentences,
        'pos_counts': pos_counts,
        'doc': None
    }


def _penn_to_universal(penn_tag):
    """
    Convert Penn Treebank POS tag to Universal Dependencies tag.
    
    Parameters
    ----------
    penn_tag : str
        Penn Treebank POS tag
    
    Returns
    -------
    str
        Universal POS tag
    """
    if penn_tag.startswith('NN'):
        return 'NOUN'
    elif penn_tag.startswith('VB'):
        return 'VERB'
    elif penn_tag.startswith('JJ'):
        return 'ADJ'
    elif penn_tag.startswith('RB'):
        return 'ADV'
    elif penn_tag.startswith('PR'):
        return 'PRON'
    elif penn_tag in ('DT', 'PDT', 'WDT'):
        return 'DET'
    elif penn_tag == 'IN':
        return 'ADP'
    elif penn_tag == 'CC':
        return 'CCONJ'
    elif penn_tag == 'CD':
        return 'NUM'
    elif penn_tag in ('UH',):
        return 'INTJ'
    elif penn_tag in ('.', ',', ':', "''", '``'):
        return 'PUNCT'
    else:
        return 'X'  # Other

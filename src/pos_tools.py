"""
POS tagging module with spaCy (preferred) and NLTK fallback.
"""

import re
from typing import Dict, List, Tuple, Optional, Iterable


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
        - pos_tokens: list of tuples (token, universal_pos, lemma)
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
    except Exception:
        pass
    
    # Fallback to NLTK
    try:
        import nltk
        return _tokenize_nltk(text)
    except ImportError:
        raise ImportError("Neither spaCy nor NLTK is available. Please install at least one.")


def tokenize_and_pos_pipe(texts: Iterable[str], batch_size: int = 32, n_process: int = 1,
                          model: str = "en_core_web_sm", disable: Optional[List[str]] = None) -> List[Dict]:
    """Batch tokenize and POS-tag a list of texts.

    Attempts spaCy pipe processing for performance; falls back to per-text processing.

    Parameters
    ----------
    texts : Iterable[str]
        Collection of input texts.
    batch_size : int, default=32
        Batch size for spaCy pipe.
    n_process : int, default=1
        Number of processes for spaCy pipe (multiprocessing). Keep 1 for determinism unless configured.
    model : str, default="en_core_web_sm"
        spaCy model name.
    disable : list of str, optional
        Pipeline components to disable for speed (e.g., ["ner"]).

    Returns
    -------
    list of dict
        List of tokenization result dicts mirroring ``tokenize_and_pos`` output.
    """
    texts = list(texts)
    if not texts:
        return []
    try:
        import spacy
        try:
            nlp = spacy.load(model, disable=disable or [])
        except OSError:
            # Model not found; try to download then load
            from spacy.cli import download
            download(model)
            nlp = spacy.load(model, disable=disable or [])

        # Ensure sentence boundaries enabled
        if "parser" not in (disable or []):
            pass  # parser provides sents

        results: List[Dict] = []
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            # Reuse internal helper logic
            results.append(_tokenize_spacy(str(doc), nlp))  # _tokenize_spacy expects text+nlp; we re-run to keep structure
        return results
    except Exception:
        # Fallback: process individually (may use spaCy single or NLTK)
        return [tokenize_and_pos(t) for t in texts]


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

    # Count POS tags and retain per-token info
    pos_counts = {}
    pos_tokens = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
        pos_tokens.append((token.text, pos, token.lemma_))
    
    return {
        'words': words,
        'sentences': sentences,
        'pos_counts': pos_counts,
        'pos_tokens': pos_tokens,
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
    from nltk.stem import WordNetLemmatizer
    
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

    # Lemmatizer for nouns/verbs/adjectives (best-effort)
    lemmatizer = WordNetLemmatizer()

    # Count POS tags (convert Penn Treebank to simplified tags)
    pos_counts = {}
    pos_tokens = []
    for word, tag in tagged:
        # Map Penn Treebank tags to Universal POS tags
        universal_tag = _penn_to_universal(tag)
        pos_counts[universal_tag] = pos_counts.get(universal_tag, 0) + 1

        # Basic lemma inference using WordNet where possible
        wn_pos = _universal_to_wordnet(universal_tag)
        lemma = lemmatizer.lemmatize(word, pos=wn_pos) if wn_pos else word
        pos_tokens.append((word, universal_tag, lemma))
    
    return {
        'words': all_words,
        'sentences': sentences,
        'pos_counts': pos_counts,
        'pos_tokens': pos_tokens,
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


def _universal_to_wordnet(universal_tag):
    """Map universal POS tags to WordNet POS tags for lemmatization."""
    try:
        from nltk.corpus import wordnet as wn  # Imported lazily to avoid hard dependency
    except ImportError:  # pragma: no cover
        return None

    mapping = {
        'NOUN': wn.NOUN,
        'VERB': wn.VERB,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV
    }

    return mapping.get(universal_tag)

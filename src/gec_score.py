"""
Grammar Error Correction Score (GECS) for AI text detection.

This module implements GECS methodology:
1. Correct grammar using GPT-4o
2. Calculate Rouge-2 similarity score
3. Higher scores indicate AI text (already grammatical)
4. Lower scores indicate human text (more corrections needed)
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from rouge import Rouge

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. GECS features will be disabled.")

# OpenAI API key from user's gecs.py
API_KEY = ""

# Initialize OpenAI client and Rouge scorer
if OPENAI_AVAILABLE:
    try:
        client = OpenAI(api_key=API_KEY)
        rouge = Rouge()
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        OPENAI_AVAILABLE = False


def correct_grammar_gpt4o(text: str, model: str = "gpt-4o-mini", temperature: float = 0.01) -> Optional[str]:
    """
    Use GPT-4o to correct grammar errors in text.
    
    Parameters
    ----------
    text : str
        Original text to correct
    model : str, default="gpt-4o-mini"
        OpenAI model to use (gpt-4o-mini is faster and cheaper)
    temperature : float, default=0.01
        Temperature for generation (low for consistency)
    
    Returns
    -------
    str or None
        Grammar-corrected text, or None if error occurs
    """
    if not OPENAI_AVAILABLE:
        logging.warning("OpenAI not available, skipping GEC")
        return None
    
    try:
        prompt = f"Correct the grammar errors in the following text: {text}\nCorrected text:"
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error during GPT-4o grammar correction: {e}")
        return None


def calculate_rouge2_score(original: str, corrected: str) -> Optional[float]:
    """
    Calculate Rouge-2 F-score between original and corrected text.
    
    Rouge-2 measures bigram overlap:
    - High score (0.95-1.0): Minimal corrections → likely AI text
    - Low score (0.7-0.9): More corrections → likely human text
    
    Parameters
    ----------
    original : str
        Original text
    corrected : str
        Grammar-corrected text
    
    Returns
    -------
    float or None
        Rouge-2 F-score (0-1), or None if error occurs
    """
    if not OPENAI_AVAILABLE or corrected is None:
        return None
    
    try:
        # Calculate Rouge scores
        scores = rouge.get_scores(original, corrected, avg=True)
        return scores['rouge-2']['f']
    
    except Exception as e:
        logging.warning(f"Failed to compute Rouge-2 score: {e}")
        return None


def compute_gecs_feature(text: str, model: str = "gpt-4o-mini") -> Dict[str, Optional[float]]:
    """
    Compute GECS feature for a single text.
    
    This is the main function to call for each document in the pipeline.
    
    Parameters
    ----------
    text : str
        Text to analyze
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'gec_text': Grammar-corrected text (str or None)
        - 'gec_rouge2_score': Rouge-2 F-score (float or None)
    """
    if not OPENAI_AVAILABLE:
        return {'gec_text': None, 'gec_rouge2_score': None}
    
    # Step 1: Correct grammar
    corrected_text = correct_grammar_gpt4o(text, model=model)
    
    # Step 2: Calculate Rouge-2 score
    rouge2_score = calculate_rouge2_score(text, corrected_text)
    
    return {
        'gec_text': corrected_text,
        'gec_rouge2_score': rouge2_score
    }


def compute_gecs_features_batch(texts: List[str], model: str = "gpt-4o-mini", 
                                verbose: bool = True) -> List[Dict[str, Optional[float]]]:
    """
    Compute GECS features for multiple texts.
    
    Parameters
    ----------
    texts : list of str
        List of texts to analyze
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    verbose : bool, default=True
        Show progress bar
    
    Returns
    -------
    list of dict
        List of result dictionaries, one per text
    """
    if not OPENAI_AVAILABLE:
        logging.warning("OpenAI not available. Returning None for all GECS features.")
        return [{'gec_text': None, 'gec_rouge2_score': None} for _ in texts]
    
    results = []
    
    if verbose:
        from tqdm import tqdm
        texts = tqdm(texts, desc="Computing GECS features")
    
    for text in texts:
        result = compute_gecs_feature(text, model=model)
        results.append(result)
    
    return results


def gecs_statistical_summary(rouge2_scores: List[float], labels: List[int]) -> Dict[str, float]:
    """
    Compute statistical summary of GECS scores by label.
    
    Parameters
    ----------
    rouge2_scores : list of float
        Rouge-2 scores for all texts
    labels : list of int
        Binary labels (0=human, 1=AI)
    
    Returns
    -------
    dict
        Statistical summary with keys:
        - 'human_mean', 'human_std': Stats for human texts
        - 'ai_mean', 'ai_std': Stats for AI texts
        - 'difference': Mean difference (AI - Human)
        - 'effect_size': Cohen's d effect size
    """
    rouge2_scores = np.array(rouge2_scores)
    labels = np.array(labels)
    
    # Filter out None values
    valid_mask = ~np.isnan(rouge2_scores)
    rouge2_scores = rouge2_scores[valid_mask]
    labels = labels[valid_mask]
    
    if len(rouge2_scores) == 0:
        return {
            'human_mean': np.nan, 'human_std': np.nan,
            'ai_mean': np.nan, 'ai_std': np.nan,
            'difference': np.nan, 'effect_size': np.nan
        }
    
    # Split by label
    human_scores = rouge2_scores[labels == 0]
    ai_scores = rouge2_scores[labels == 1]
    
    # Compute statistics
    human_mean = np.mean(human_scores) if len(human_scores) > 0 else np.nan
    human_std = np.std(human_scores) if len(human_scores) > 0 else np.nan
    ai_mean = np.mean(ai_scores) if len(ai_scores) > 0 else np.nan
    ai_std = np.std(ai_scores) if len(ai_scores) > 0 else np.nan
    
    # Effect size (Cohen's d)
    if len(human_scores) > 0 and len(ai_scores) > 0:
        pooled_std = np.sqrt(((len(human_scores)-1)*human_std**2 + 
                              (len(ai_scores)-1)*ai_std**2) / 
                             (len(human_scores) + len(ai_scores) - 2))
        effect_size = (ai_mean - human_mean) / pooled_std if pooled_std > 0 else 0.0
    else:
        effect_size = np.nan
    
    return {
        'human_mean': human_mean,
        'human_std': human_std,
        'ai_mean': ai_mean,
        'ai_std': ai_std,
        'difference': ai_mean - human_mean,
        'effect_size': effect_size
    }

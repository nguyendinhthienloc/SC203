"""
Grammar Error Correction Score (GECS) for AI text detection.

This module implements GECS methodology:
1. Correct grammar using GPT-4o
2. Calculate Rouge-2 similarity score
3. Higher scores indicate AI text (already grammatical)
4. Lower scores indicate human text (more corrections needed)

Can be used as a standalone script:
- With CSV input: python gec_score.py --csv input.csv --text-col text
- Interactive mode: python gec_score.py (prompts for text input)

Environment Variables:
- OPENAI_API_KEY: Required for GPT-4o access (set in .env file)
"""

import logging
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from rouge import Rouge

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv not installed. Install with: pip install python-dotenv")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. GECS features will be disabled.")

# Load OpenAI API key from environment variable
API_KEY = os.getenv("API_KEY", "")

# Initialize OpenAI client and Rouge scorer
if OPENAI_AVAILABLE:
    try:
        if not API_KEY:
            logging.warning("OPENAI_API_KEY not set in environment or .env file")
            OPENAI_AVAILABLE = False
        else:
            client = OpenAI(api_key=API_KEY)
            rouge = Rouge()
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        OPENAI_AVAILABLE = False


def correct_grammar_gpt4o(text: str, model: str = "gpt-5-mini", temperature: float = 0.01, 
                          use_ai_prompt: bool = True) -> Optional[str]:
    """
    Use GPT-4o to correct grammar errors in text or prompt AI to generate content.
    
    Parameters
    ----------
    text : str
        Original text to correct or use as prompt
    model : str, default="gpt-4o-mini"
        OpenAI model to use (gpt-4o-mini is faster and cheaper)
    temperature : float, default=0.01
        Temperature for generation (low for consistency)
    use_ai_prompt : bool, default=True
        If True, uses text as a prompt for AI generation (default for standalone use)
        If False, directly corrects grammar (for pipeline integration)
    
    Returns
    -------
    str or None
        Grammar-corrected text or AI-generated response, or None if error occurs
    """
    if not OPENAI_AVAILABLE:
        logging.warning("OpenAI not available, skipping GEC")
        return None
    
    try:
        if use_ai_prompt:
            # Prompt mode: Use the text as a query/prompt for AI
            prompt = text
        else:
            # Grammar correction mode: Explicitly ask to correct grammar
            prompt = f"Correct the grammar errors in the following text: {text}\nCorrected text:"
        
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        
        return completion.choices[0].message.content
    
    except Exception as e:
        logging.error(f"Error during GPT-4o processing: {e}")
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


def compute_gecs_feature(text: str, model: str = "gpt-4o-mini", 
                        use_ai_prompt: bool = False) -> Dict[str, Optional[float]]:
    """
    Compute GECS feature for a single text.
    
    This is the main function to call for each document in the pipeline.
    
    Parameters
    ----------
    text : str
        Text to analyze
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    use_ai_prompt : bool, default=False
        If True, uses text as AI prompt (for interactive mode)
        If False, corrects grammar directly (for pipeline, default)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'gec_text': Grammar-corrected text (str or None)
        - 'gec_rouge2_score': Rouge-2 F-score (float or None)
    """
    if not OPENAI_AVAILABLE:
        return {'gec_text': None, 'gec_rouge2_score': None}
    
    # Step 1: Correct grammar or prompt AI
    corrected_text = correct_grammar_gpt4o(text, model=model, use_ai_prompt=use_ai_prompt)
    
    # Step 2: Calculate Rouge-2 score
    rouge2_score = calculate_rouge2_score(text, corrected_text)
    
    return {
        'gec_text': corrected_text,
        'gec_rouge2_score': rouge2_score
    }


def compute_gecs_features_batch(texts: List[str], model: str = "gpt-4o-mini", 
                                verbose: bool = True, use_ai_prompt: bool = False) -> List[Dict[str, Optional[float]]]:
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
    use_ai_prompt : bool, default=False
        If True, uses texts as AI prompts (for interactive mode)
        If False, corrects grammar directly (for pipeline, default)
    
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
        result = compute_gecs_feature(text, model=model, use_ai_prompt=use_ai_prompt)
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


def process_csv_input(csv_path: str, text_col: str = "text", output_path: str = None,
                      model: str = "gpt-4o-mini", use_ai_prompt: bool = False) -> None:
    """
    Process texts from CSV file and compute GECS scores.
    
    Parameters
    ----------
    csv_path : str
        Path to input CSV file
    text_col : str, default="text"
        Column name containing text to analyze
    output_path : str, optional
        Path to save results CSV (default: input_path_gecs.csv)
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    use_ai_prompt : bool, default=False
        If True, uses texts as AI prompts
        If False, corrects grammar directly (default)
    """
    import pandas as pd
    
    logging.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if text_col not in df.columns:
        logging.error(f"Column '{text_col}' not found in CSV. Available columns: {list(df.columns)}")
        sys.exit(1)
    
    logging.info(f"Processing {len(df)} texts...")
    texts = df[text_col].astype(str).tolist()
    
    # Compute GECS features
    results = compute_gecs_features_batch(texts, model=model, verbose=True, use_ai_prompt=use_ai_prompt)
    
    # Add results to dataframe
    df['gec_text'] = [r['gec_text'] for r in results]
    df['gec_rouge2_score'] = [r['gec_rouge2_score'] for r in results]
    
    # Save results
    if output_path is None:
        output_path = str(Path(csv_path).stem) + "_gecs.csv"
    
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    
    # Print summary statistics
    valid_scores = [s for s in df['gec_rouge2_score'] if s is not None and not pd.isna(s)]
    if valid_scores:
        logging.info(f"\n{'='*50}")
        logging.info("GECS Summary Statistics:")
        logging.info(f"  Processed: {len(valid_scores)}/{len(df)} texts")
        logging.info(f"  Mean Rouge-2 score: {np.mean(valid_scores):.4f}")
        logging.info(f"  Std Rouge-2 score:  {np.std(valid_scores):.4f}")
        logging.info(f"  Min Rouge-2 score:  {np.min(valid_scores):.4f}")
        logging.info(f"  Max Rouge-2 score:  {np.max(valid_scores):.4f}")
        logging.info(f"{'='*50}")


def process_interactive_mode(model: str = "gpt-4o-mini") -> None:
    """
    Interactive mode: prompt user for text and compute GECS score.
    
    Parameters
    ----------
    model : str, default="gpt-4o-mini"
        OpenAI model to use
    """
    print("\n" + "="*60)
    print("GECS Interactive Mode - Grammar Error Correction Score")
    print("="*60)
    print("\nThis tool will:")
    print("1. Correct grammar in your text using GPT-4o")
    print("2. Calculate Rouge-2 similarity score")
    print("3. Higher scores suggest AI-generated text")
    print("4. Lower scores suggest human-written text")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        print("-" * 60)
        text = input("Enter text to analyze (or 'quit' to exit):\n> ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nExiting GECS analyzer. Goodbye!")
            break
        
        if not text.strip():
            print("⚠️  Empty input. Please enter some text.")
            continue
        
        print("\n🔄 Processing with GPT-4o...")
        result = compute_gecs_feature(text, model=model, use_ai_prompt=True)
        
        corrected = result['gec_text']
        score = result['gec_rouge2_score']
        
        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        
        if corrected is None or score is None:
            print("❌ Error: Could not process text. Check API key and connection.")
        else:
            print(f"\n📝 Original text:")
            print(f"   {text[:200]}{'...' if len(text) > 200 else ''}")
            
            print(f"\n✅ Corrected text:")
            print(f"   {corrected[:200]}{'...' if len(corrected) > 200 else ''}")
            
            print(f"\n📊 GECS Score (Rouge-2): {score:.4f}")
            
            # Interpretation
            if score >= 0.95:
                interpretation = "Very High - Likely AI-generated (minimal corrections)"
            elif score >= 0.85:
                interpretation = "High - Possibly AI-generated"
            elif score >= 0.75:
                interpretation = "Moderate - Mixed characteristics"
            else:
                interpretation = "Low - Likely human-written (more corrections needed)"
            
            print(f"💡 Interpretation: {interpretation}")
        
        print("\n")


def main():
    """Main entry point for standalone script usage."""
    parser = argparse.ArgumentParser(
        description="Grammar Error Correction Score (GECS) for AI text detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python gec_score.py
  
  # Process CSV file
  python gec_score.py --csv data/texts.csv --text-col content
  
  # Process CSV with custom output and model
  python gec_score.py --csv input.csv --output results.csv --model gpt-4o
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to input CSV file (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--text-col',
        type=str,
        default='text',
        help='Column name containing text to analyze (default: "text")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file (default: input_path_gecs.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model to use (default: "gpt-4o-mini")'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--prompt-mode',
        action='store_true',
        help='Use text as AI prompt instead of grammar correction (default for interactive)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        logging.error("OpenAI package not installed. Please run: pip install openai")
        sys.exit(1)
    
    if not API_KEY or API_KEY == "":
        logging.error("OpenAI API key not configured. Please set API_KEY in gec_score.py")
        sys.exit(1)
    
    # Determine mode based on --csv flag
    if args.csv:
        # CSV processing mode
        # For CSV, use prompt mode only if explicitly requested
        process_csv_input(
            csv_path=args.csv,
            text_col=args.text_col,
            output_path=args.output,
            model=args.model,
            use_ai_prompt=args.prompt_mode
        )
    else:
        # Interactive mode (always uses prompt mode by default)
        process_interactive_mode(model=args.model)


if __name__ == "__main__":
    main()

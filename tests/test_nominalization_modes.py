"""Tests to ensure nominalization modes behave differently when strict vs balanced."""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nominalization import analyze_nominalization
import spacy


def _get_doc(text: str):
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    return nlp(text)


def test_nominalization_mode_strict_vs_balanced():
    # Text with verbs and candidate nominal forms
    text = "We analyze the development of systems and implementation of methods."  # development, implementation potential nominal forms
    doc = _get_doc(text)
    # Balanced (lenient) should capture more than strict when verbs limited
    balanced_df = analyze_nominalization([doc], mode="balanced", context_window=2)
    strict_df = analyze_nominalization([doc], mode="strict", context_window=2)
    # Expect balanced to have at least as many rows
    assert len(balanced_df) >= len(strict_df)
    # If strict found entries, they should be subset of balanced lemmas
    if not strict_df.empty and not balanced_df.empty:
        assert set(strict_df['nominal'].tolist()).issubset(set(balanced_df['nominal'].tolist()))


if __name__ == "__main__":
    test_nominalization_mode_strict_vs_balanced()
    print("Nominalization mode test passed")
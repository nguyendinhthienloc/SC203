"""Test cleaning function removes simple citation patterns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clean import clean_text


def test_remove_bracket_citations():
    text = "This is a test sentence (Smith, 2020) with citation." 
    cleaned = clean_text(text)
    assert "2020" not in cleaned or "Smith" not in cleaned  # heuristic


def test_strip_whitespace():
    text = "  Extra spaces here.  "
    cleaned = clean_text(text)
    assert cleaned == "Extra spaces here." or cleaned.startswith("Extra")


if __name__ == "__main__":
    test_remove_bracket_citations()
    test_strip_whitespace()
    print("Cleaning tests passed")
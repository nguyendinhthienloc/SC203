"""
Unit tests for nominalization detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nominalization import detect_nominals_suffix, detect_nominals_spacy


def test_suffix_detector():
    """Test suffix-based nominalization detection."""
    tokens = ["development", "implementation", "run", "analysis", "happiness", 
              "creation", "decision", "government", "reference", "action"]
    
    result = detect_nominals_suffix(tokens)
    
    # Should detect multiple nominalizations
    assert result["nominal_from_verb"] >= 2, \
        f"Expected at least 2 nominalizations, got {result['nominal_from_verb']}"
    
    # Check specific suffixes
    assert result["suffix_counts"]["ment"] >= 2, \
        f"Expected at least 2 '-ment' nominalizations"
    
    assert result["suffix_counts"]["tion"] >= 1, \
        f"Expected at least 1 '-tion' nominalization"
    
    print("✓ Suffix detector test passed")


def test_suffix_detector_empty():
    """Test suffix detector with empty input."""
    result = detect_nominals_suffix([])
    
    assert result["nominal_from_verb"] == 0
    assert result["all_noun_count"] == 0
    
    print("✓ Empty input test passed")


def test_suffix_detector_no_nominals():
    """Test suffix detector with no nominalizations."""
    tokens = ["cat", "dog", "run", "jump", "quick", "brown"]
    
    result = detect_nominals_suffix(tokens)
    
    # Should detect zero or very few
    assert result["nominal_from_verb"] <= 1, \
        f"Expected 0-1 nominalizations, got {result['nominal_from_verb']}"
    
    print("✓ No nominals test passed")


def test_spacy_detector():
    """Test spaCy lemma-based detection if spaCy is available."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⊘ Skipping spaCy test (model not installed)")
            return
        
        text = "The development of the system requires implementation. We need to develop better implementations."
        doc = nlp(text)
        
        result = detect_nominals_spacy(doc)
        
        # Should detect at least some nominalizations
        assert result["all_noun_count"] > 0, "Should find some nouns"
        assert isinstance(result["nominal_from_verb"], int)
        
        print(f"✓ spaCy detector test passed (found {result['nominal_from_verb']} nominalizations)")
        
    except ImportError:
        print("⊘ Skipping spaCy test (not installed)")


def run_all_tests():
    """Run all nominalization tests."""
    print("Running nominalization tests...\n")
    
    test_suffix_detector()
    test_suffix_detector_empty()
    test_suffix_detector_no_nominals()
    test_spacy_detector()
    
    print("\n" + "=" * 50)
    print("All nominalization tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()

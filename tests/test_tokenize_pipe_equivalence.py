"""Ensure batched tokenize_and_pos_pipe yields comparable results to single calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pos_tools import tokenize_and_pos, tokenize_and_pos_pipe


def test_pipe_equivalence_basic():
    texts = ["Simple test sentence.", "Another sentence for testing."]
    single = [tokenize_and_pos(t) for t in texts]
    batched = tokenize_and_pos_pipe(texts, batch_size=2, n_process=1)
    assert len(single) == len(batched)
    # Compare word counts
    for s, b in zip(single, batched):
        assert len(s['words']) == len(b['words'])


if __name__ == "__main__":
    test_pipe_equivalence_basic()
    print("Pipe equivalence test passed")
import pytest
import sys
import os

# Add repo root to sys.path so 'tools' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.ngram_sim import simulate_turn, TokenizerWrapper, process_file
import io

def test_simulate_turn():
    context = ["A", "B", "C", "A", "B", "C"]
    target = ["A", "B", "C", "D"]

    tot, acc, steps, tot_220, acc_220, steps_220 = simulate_turn(context, target, n=2, k=4)
    assert tot == 4
    assert acc == 3
    assert steps == 1
    assert tot_220 == 4
    assert acc_220 == 3
    assert steps_220 == 1

def test_simulate_turn_step_by_step():
    context = ["X", "Y", "Z", "X", "Y", "Z", "W", "X", "Y", "Z"]
    target = ["W", "Q"]

    tot, acc, steps, tot_220, acc_220, steps_220 = simulate_turn(context, target, n=3, k=2)
    assert tot == 2
    assert acc == 1
    assert steps == 1

def test_tokenizer_fallback():
    tok = TokenizerWrapper()
    tok.is_fallback = True  # force fallback for test
    tokens = tok.encode("a b,c")
    assert tokens == ["a", " ", "b", ",", "c"]

def test_fixture_end_to_end(capsys):
    import tools.ngram_sim as sim
    original_has_transformers = sim.HAS_TRANSFORMERS
    sim.HAS_TRANSFORMERS = False

    fixture_path = os.path.join(os.path.dirname(__file__), 'fixtures', 'ngram_synthetic.jsonl')

    try:
        process_file(fixture_path, n=2, k=4, limit=None)
        captured = capsys.readouterr()

        # Test exact manual expected acceptance values!
        assert "Total Output Tokens            | 16" in captured.out
        assert "Total Accepted Draft Tokens    | 8" in captured.out
        assert "Fraction Covered               | 0.5000" in captured.out
    finally:
        sim.HAS_TRANSFORMERS = original_has_transformers

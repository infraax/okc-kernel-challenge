import pytest
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.draft_cascade import DraftCascadeSimulator, TokenizerWrapper, load_dataset_local

def test_cascade_simulator():
    tokenizer = TokenizerWrapper()
    sim = DraftCascadeSimulator(tau=1, dflash_mean=4.2)
    
    data = [
        [
            {"role": "user", "content": "A B C"},
            {"role": "assistant", "content": "A B C D E"}
        ]
    ]
    os.makedirs("tests/fixtures", exist_ok=True)
    with open("tests/fixtures/cascade_synthetic.jsonl", "w") as f:
        f.write(json.dumps(data[0]) + "\n")
        
    convs = load_dataset_local("tests/fixtures/cascade_synthetic.jsonl")
    assert len(convs) == 1
    
    # We will pass a clean list of string tokens for testing
    context_tokens = ["A", "B", "C", "A", "B", "C"]
    target_tokens = ["D", "E", "F"]
    
    res = sim.simulate_turn(context_tokens, target_tokens, 'ngram')
    
    assert res['total_tokens'] == 3
    assert res['accepted_tokens'] == 0
    assert res['steps'] == 3
    
    # Test a match: 
    # context ends with "X", "Y"
    # earlier in context we have "X", "Y", "Z", "W"
    # target is "Z", "W", "Q"
    # With n=2, prefix is ["X", "Y"], match gives continuation ["Z", "W"]
    # Target starts with ["Z", "W"], so we should accept 2 tokens!
    context_tokens = ["X", "Y", "Z", "W", "X", "Y"]
    target_tokens = ["Z", "W", "Q"]
    res = sim.simulate_turn(context_tokens, target_tokens, 'ngram')
    assert res['total_tokens'] == 3
    # Step 1: draft Z W, verified -> accepts 2 (Z, W). Then auto-regressive token is Q.
    # The step advanced by m+1 = 3. Total idx is 3. We are done in 1 step!
    assert res['accepted_tokens'] == 2
    assert res['steps'] == 1

#!/usr/bin/env python3
"""
Simulate prompt-lookup n-gram drafting on multi-turn agentic coding transcripts.

This tool evaluates how many output tokens of agentic-coding turns can be drafted
from the same conversation's earlier tokens using a prompt-lookup mechanism.

Dataset Loader:
This script supports JSONL transcripts where each line is a list of turn objects:
`[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]`.
A suitable dataset is OpenHands trajectories or SWE-agent trajectories from Hugging Face:
- `HuggingFaceFW/SWE-agent-trajectories` (requires converting to JSONL locally, or writing a custom fetcher, but expecting a local JSONL for this script)
- Actually, any JSONL with `[{"role": "...", "content": "..."}, ...]` format.

Usage:
  python tools/ngram_sim.py path/to/dataset.jsonl [--n 3] [--k 8] [--limit 100]

Dependencies:
  - `transformers` (optional, uses Llama-class tokenizer if available)
"""
import argparse
import json
import logging
import re
import sys
from typing import List, Dict, Any, Tuple, Optional

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class TokenizerWrapper:
    def __init__(self):
        self.tokenizer = None
        self.is_fallback = True

        if HAS_TRANSFORMERS:
            try:
                # Try to load a local/public Llama tokenizer.
                # If unavailable offline, fallback is triggered.
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-2-7b-hf", local_files_only=True
                )
                self.is_fallback = False
                logging.info("Loaded Llama tokenizer successfully.")
            except Exception as e:
                logging.warning(f"Could not load Hugging Face tokenizer ({e}). Using fallback.")
        else:
            logging.info("transformers not installed. Using fallback tokenizer.")

    def encode(self, text: str) -> List[str]:
        if not self.is_fallback and self.tokenizer is not None:
            # We map back to string tokens or keep as IDs?
            # Keeping as IDs is faster, but strings work for fallback.
            # We will use integer IDs for transformers, and string words for fallback.
            # Let's standardize on whatever encode returns:
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            # Fallback: whitespace + punctuation
            return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|\s+|[^\w\s]', text)

def simulate_turn(
    context_tokens: List[Any],
    target_tokens: List[Any],
    n: int,
    k: int
) -> Tuple[int, int, int, int, int, int]:
    """
    Simulate draft acceptance for a single turn.

    Args:
        context_tokens: Tokens from previous turns + generated prefix in this turn.
        target_tokens: The tokens the assistant will generate in this turn.
        n: The n-gram prefix size to match.
        k: The draft length to propose.

    Returns:
        total_tokens: length of target_tokens
        accepted_draft_tokens: total number of draft tokens accepted
        steps: number of auto-regressive / draft-verification steps taken
        total_tokens_220: min(total_tokens, 220)
        accepted_draft_tokens_220: accepted draft tokens within the first 220 output tokens
        steps_220: steps taken within the first 220 tokens
    """
    total_tokens = len(target_tokens)
    accepted_draft_tokens = 0
    accepted_draft_tokens_220 = 0
    steps = 0
    steps_220 = 0

    idx = 0
    # context is a copy so we can append to it
    current_context = list(context_tokens)

    while idx < total_tokens:
        steps += 1
        if idx < 220:
            steps_220 += 1

        # Propose draft
        draft = []
        if len(current_context) >= n:
            ngram_prefix = current_context[-n:]

            # Find the *last* occurrence of this n-gram in the context
            match_idx = -1
            search_space_len = len(current_context) - n
            for i in range(search_space_len - 1, -1, -1):
                if current_context[i:i+n] == ngram_prefix:
                    match_idx = i
                    break

            if match_idx != -1:
                # We found a match, propose up to k tokens following the match
                start_draft = match_idx + n
                draft = current_context[start_draft : start_draft + k]

        # Verify draft against target_tokens[idx : idx + 1 + len(draft)]
        m = 0
        for i in range(len(draft)):
            if idx + i < total_tokens and draft[i] == target_tokens[idx + i]:
                m += 1
            else:
                break

        accepted_draft_tokens += m

        # If idx < 220, track accepted tokens in the first 220
        for i in range(m):
            if idx + i < 220:
                accepted_draft_tokens_220 += 1

        # The step advances by m (accepted draft tokens) + 1 (the auto-regressive token)
        advance = m + 1

        # Add the generated tokens to the context
        current_context.extend(target_tokens[idx : idx + advance])

        idx += advance

    total_220 = min(total_tokens, 220)
    return total_tokens, accepted_draft_tokens, steps, total_220, accepted_draft_tokens_220, steps_220

def process_file(filepath: str, n: int, k: int, limit: Optional[int]) -> None:
    tokenizer = TokenizerWrapper()

    total_out_tokens = 0
    total_acc_tokens = 0
    total_steps = 0

    total_out_tokens_220 = 0
    total_acc_tokens_220 = 0
    total_steps_220 = 0

    processed_conversations = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            if limit is not None and processed_conversations >= limit:
                break

            conversation = json.loads(line)
            context_tokens = []

            for turn in conversation:
                role = turn.get("role")
                content = turn.get("content", "")
                turn_tokens = tokenizer.encode(content)

                if role == "assistant":
                    if not turn_tokens:
                        continue

                    t_tot, t_acc, t_step, t_tot_220, t_acc_220, t_step_220 = simulate_turn(
                        context_tokens, turn_tokens, n, k
                    )

                    total_out_tokens += t_tot
                    total_acc_tokens += t_acc
                    total_steps += t_step

                    total_out_tokens_220 += t_tot_220
                    total_acc_tokens_220 += t_acc_220
                    total_steps_220 += t_step_220

                # After processing (or if it's a user turn), add to context
                context_tokens.extend(turn_tokens)

            processed_conversations += 1

    if total_out_tokens == 0:
        print("No assistant tokens found.")
        return

    frac_acc = total_acc_tokens / total_out_tokens
    mean_acc_len = total_acc_tokens / total_steps if total_steps > 0 else 0.0

    frac_acc_220 = total_acc_tokens_220 / total_out_tokens_220 if total_out_tokens_220 > 0 else 0.0
    mean_acc_len_220 = total_acc_tokens_220 / total_steps_220 if total_steps_220 > 0 else 0.0

    print(f"{'Metric':<30} | {'All Tokens':<15} | {'First 220 Tokens':<15}")
    print("-" * 66)
    print(f"{'Total Output Tokens':<30} | {total_out_tokens:<15} | {total_out_tokens_220:<15}")
    print(f"{'Total Accepted Draft Tokens':<30} | {total_acc_tokens:<15} | {total_acc_tokens_220:<15}")
    print(f"{'Fraction Covered':<30} | {frac_acc:<15.4f} | {frac_acc_220:<15.4f}")
    print(f"{'Mean Accepted Length / Step':<30} | {mean_acc_len:<15.4f} | {mean_acc_len_220:<15.4f}")
    print()
    print(f"Summary: n={n}, k={k} | Covered: {frac_acc:.1%} (All), {frac_acc_220:.1%} (First 220) | Mean acc/step: {mean_acc_len:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Simulate prompt-lookup n-gram drafting.")
    parser.add_argument("jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--n", type=int, default=3, choices=[2, 3, 4, 5], help="n-gram prefix size")
    parser.add_argument("--k", type=int, default=8, choices=[4, 8], help="draft length")
    parser.add_argument("--limit", type=int, default=None, help="limit number of conversations to process")

    args = parser.parse_args()

    process_file(args.jsonl, args.n, args.k, args.limit)

if __name__ == "__main__":
    main()

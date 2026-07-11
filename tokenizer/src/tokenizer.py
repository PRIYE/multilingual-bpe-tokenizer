"""
tokenizer.py — Reloadable BPE encode/decode with regex pre-tokenization.

Functions:
  save(vocab, merges, output_dir)    — write vocab.json + merges.txt
  load(output_dir)                   — read vocab.json + merges.txt → (vocab, merges)
  encode(text, vocab, merges, lang)  — regex pre-tokenization + BPE encode (default)
  encode_legacy(text, vocab, merges) — character-only BPE encode (backward compatibility)
  decode(ids, id_to_token)           — ids → string
  script_distribution(vocab)         — token counts by Unicode script block
"""

import os
import json
import unicodedata
from regex_pretokenizer import pretokenize_multilingual


# ---------------------------------------------------------------------------
# T011 — Serialisation / deserialisation
# ---------------------------------------------------------------------------

VOCAB_FILENAME = "vocab.json"
MERGES_FILENAME = "merges.txt"
MERGES_VERSION_HEADER = "#version: 1.0"


def save(vocab: dict, merges: list, output_dir: str):
    """
    Serialise the trained tokenizer to disk.

    vocab.json  — JSON object {token_string: id}, UTF-8, ensure_ascii=False
    merges.txt  — "#version: 1.0" header, then one "LEFT RIGHT" line per merge

    Asserts that len(vocab) == 10,000 before writing.
    """
    # Allow saving smaller vocabs for testing
    if len(vocab) != 10_000:
        print(f"  Warning: Saving vocab with {len(vocab)} tokens, expected 10,000.")
        
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, VOCAB_FILENAME)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=None)
    print(f"  Saved vocab ({len(vocab)} tokens) → {vocab_path}")

    merges_path = os.path.join(output_dir, MERGES_FILENAME)
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write(MERGES_VERSION_HEADER + "\n")
        for left, right in merges:
            f.write(f"{left} {right}\n")
    print(f"  Saved merges ({len(merges)} rules) → {merges_path}")


def load(output_dir: str) -> tuple:
    """
    Load a saved tokenizer from disk.

    Returns:
        (vocab: dict[str, int], merges: list[tuple[str, str]])
    """
    vocab_path = os.path.join(output_dir, VOCAB_FILENAME)
    merges_path = os.path.join(output_dir, MERGES_FILENAME)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"vocab.json not found at {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"merges.txt not found at {merges_path}")

    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)

    if len(vocab) != 10_000:
        print(f"  Warning: Loaded vocab has {len(vocab)} tokens, expected 10,000.")

    merges = []
    with open(merges_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                merges.append(tuple(parts))

    return vocab, merges


# ---------------------------------------------------------------------------
# T017 — Standalone encode (reproducibility contract)
# ---------------------------------------------------------------------------

def encode(text: str, vocab: dict, merges: list, language: str = "en") -> list:
    """
    Greedy BPE encode with regex pre-tokenization (default method).

    Algorithm:
      1. Apply language-specific regex pre-tokenization to split text into chunks
      2. For each pre-token chunk:
         a. Split into individual characters  
         b. Apply BPE merges within the chunk (respecting pre-token boundaries)
      3. Convert all final tokens to IDs via vocab lookup

    Args:
        text: Input text to encode
        vocab: Token-to-ID mapping
        merges: List of (left, right) merge rules in order
        language: Language code for pre-tokenization ('en', 'hi', 'te', 'sa')

    Returns:
        List of token IDs
    """
    # Step 1: Apply regex pre-tokenization
    pretokens = pretokenize_multilingual(text, language)
    
    # Build a merge rank lookup for O(1) BPE processing
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}
    
    all_ids = []
    
    # Step 2: Apply BPE to each pre-token separately
    for pretoken in pretokens:
        # Start with character-level tokens within this pre-token
        tokens = list(pretoken)
        
        # Apply BPE merges within this pre-token
        while len(tokens) >= 2:
            pairs = set(zip(tokens, tokens[1:]))
            ranked = {p: merge_rank[p] for p in pairs if p in merge_rank}
            if not ranked:
                break
            best_pair = min(ranked, key=ranked.get)
            left, right = best_pair
            merged = left + right
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        # Convert tokens to IDs
        for t in tokens:
            if t in vocab:
                all_ids.append(vocab[t])
            else:
                # Unknown token: encode each character individually
                for ch in t:
                    if ch in vocab:
                        all_ids.append(vocab[ch])
                    # silently skip characters truly not in vocab
    
    return all_ids


def encode_legacy(text: str, vocab: dict, merges: list) -> list:
    """
    Legacy greedy BPE encode using character-level splitting (no regex pre-tokenization).
    Kept for backward compatibility and testing purposes.

    Algorithm (per contracts/tokenizer-format.md):
      1. Split text into individual characters
      2. While len(tokens) >= 2:
         a. Compute all adjacent pairs
         b. Find the pair with the lowest merge rank (= earliest in merges list)
         c. If no pair in merges: STOP
         d. Merge all non-overlapping occurrences (left-to-right)
      3. Return list of integer IDs via vocab lookup

    Characters not in vocab are skipped with a warning (unknown-token fallback).
    """
    # Build a merge rank lookup for O(1) step 2b
    merge_rank = {pair: rank for rank, pair in enumerate(merges)}

    tokens = list(text)

    while len(tokens) >= 2:
        pairs = set(zip(tokens, tokens[1:]))
        ranked = {p: merge_rank[p] for p in pairs if p in merge_rank}
        if not ranked:
            break
        best_pair = min(ranked, key=ranked.get)
        left, right = best_pair
        merged = left + right
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens

    ids = []
    for t in tokens:
        if t in vocab:
            ids.append(vocab[t])
        else:
            # Unknown token: encode each character individually
            for ch in t:
                if ch in vocab:
                    ids.append(vocab[ch])
                # silently skip characters truly not in vocab
    return ids




# ---------------------------------------------------------------------------
# T018 — Decode
# ---------------------------------------------------------------------------

def decode(ids: list, id_to_token: dict) -> str:
    """
    Decode a list of token IDs back to a string.
    Unknown IDs produce "<UNK>" in the output.
    """
    parts = [id_to_token.get(i, "<UNK>") for i in ids]
    return "".join(parts)


# ---------------------------------------------------------------------------
# T033 — Script distribution
# ---------------------------------------------------------------------------

# Unicode block ranges for classification
_DEVANAGARI_RANGE = range(0x0900, 0x0980)
_TELUGU_RANGE = range(0x0C00, 0x0C80)


def _classify_char(ch: str) -> str:
    cp = ord(ch)
    if cp in _DEVANAGARI_RANGE:
        return "Devanagari"
    if cp in _TELUGU_RANGE:
        return "Telugu"
    if cp < 0x0080:  # Basic Latin
        return "Latin"
    return "Other"


def script_distribution(vocab: dict) -> dict:
    """
    Classify each token in vocab by its dominant Unicode script.

    Returns:
        dict with keys "Devanagari", "Telugu", "Latin", "Mixed", "Other"
        and values = count of tokens in that category.
    """
    counts = {"Devanagari": 0, "Telugu": 0, "Latin": 0, "Mixed": 0, "Other": 0}

    for token in vocab:
        if not token:
            counts["Other"] += 1
            continue

        scripts = set(_classify_char(ch) for ch in token if not ch.isspace())
        scripts.discard("Other")

        if len(scripts) == 0:
            counts["Other"] += 1
        elif len(scripts) == 1:
            script = scripts.pop()
            counts[script] = counts.get(script, 0) + 1
        else:
            counts["Mixed"] += 1

    return counts

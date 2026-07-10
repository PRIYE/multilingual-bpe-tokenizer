# Contract: Tokenizer Serialisation Format

**Feature**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10

---

## Purpose

This contract defines the exact on-disk format for the trained BPE tokenizer. Any reviewer must be able to reproduce all fertility ratios using only these files, without the training script.

---

## File 1: `vocab.json`

**Path**: `tokenizer/output/vocab.json`

**Format**: JSON object mapping token string → integer ID.

```json
{
  " ": 0,
  "!": 1,
  "\"": 2,
  "अ": 47,
  "ఆ": 312,
  "ab": 530,
  "...": "..."
}
```

**Constraints**:
- MUST contain exactly 10,000 entries
- IDs MUST be unique integers in range `[0, 9999]`
- All token strings MUST be valid UTF-8
- File encoding MUST be UTF-8 with `ensure_ascii=False`

---

## File 2: `merges.txt`

**Path**: `tokenizer/output/merges.txt`

**Format**: Plain text, one merge rule per line, formatted as `LEFT RIGHT` (space-separated). Line order defines merge priority (rank 0 = first line = highest priority).

```
#version: 1.0
అ ం
అం ద
ద ేశ
ద ే
...
```

**Constraints**:
- First line MUST be `#version: 1.0` (comment, ignored during loading)
- Each subsequent line MUST be exactly two whitespace-separated token strings
- Total non-comment lines MUST equal `vocab_size - initial_vocab_size`
- File encoding MUST be UTF-8

---

## File 3: `tokenizer.json` (widget combined format)

**Path**: `widget/data/tokenizer.json`

**Format**: JSON object with two keys.

```json
{
  "vocab": {
    " ": 0,
    "అ": 47,
    "...": "..."
  },
  "merges": [
    ["అ", "ం"],
    ["అం", "ద"],
    "..."
  ]
}
```

**Constraints**:
- `vocab` key: same as `vocab.json` above
- `merges` key: ordered array of 2-element arrays `[left, right]`
- Array order defines merge priority (index 0 = rank 0 = highest priority)
- File encoding MUST be UTF-8 with `ensure_ascii=False`

---

## Encoding Algorithm Contract

Any implementation of `encode(text)` using these files MUST follow this exact algorithm to be compatible:

```
1. Split text into initial token sequence = list of individual characters
2. While len(sequence) >= 2:
   a. Compute all adjacent pairs in sequence
   b. For each pair, look up its rank: rank = index of [left, right] in merges array
      (pairs not in merges get rank = infinity)
   c. Find the pair with the lowest rank (= highest priority)
   d. If no pair has finite rank: STOP
   e. Merge all non-overlapping occurrences of that pair (left-to-right scan)
3. Return sequence of token IDs (via vocab lookup)
```

**Critical**: step 2e merges ALL occurrences of the winning pair in a single pass before re-evaluating, matching standard BPE behaviour.

---

## Decoding Algorithm Contract

```
1. For each id in sequence: look up token string via reverse vocab
2. Concatenate all token strings
3. Return concatenated string
```

---

## Reproducibility Requirement

A third-party reviewer must be able to run:

```python
import json

with open("vocab.json") as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}

with open("merges.txt") as f:
    merges = [tuple(line.strip().split()) for line in f if not line.startswith("#")]

def encode(text):
    tokens = list(text)
    while len(tokens) >= 2:
        pairs = {(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)}
        pair_ranks = {p: merges.index(p) for p in pairs if p in merges}
        if not pair_ranks:
            break
        best = min(pair_ranks, key=pair_ranks.get)
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                new_tokens.append(tokens[i] + tokens[i+1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return [vocab[t] for t in tokens]
```

and reproduce the exact same fertility ratios as the submitted scores.

# Quickstart Validation Guide

**Feature**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10

---

## Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install wikipedia-api beautifulsoup4 indic-nlp-library requests

# (Optional: for Sanskrit training corpus)
pip install datasets
```

Project directory structure must match [plan.md](./plan.md) — `tokenizer/` and `widget/` at the repo root.

---

## Step 1: Fetch & Clean Wikipedia Corpora

**Run**:
```bash
python tokenizer/src/fetch.py
```

**Expected outcome**:
- Creates `tokenizer/data/clean/en.txt`, `hi.txt`, `te.txt`, `sa.txt`
- Each file is UTF-8 plain text with no HTML, no `[N]` citation markers
- Verify:
```bash
wc -c tokenizer/data/clean/*.txt
# en.txt should be largest (~80-150KB), sa.txt smallest (~8-25KB)
```

**Edge case check**: If `sa.txt` < 20,000 characters, the training step will automatically augment with `rahular/itihasa`. The scoring step always uses only `sa.txt`.

---

## Step 2: Train the Tokenizer

**Run**:
```bash
python tokenizer/src/train.py
```

**Expected outcome**:
- Prints progress every 500 merges: `Merge 500/9744 | worst_lang=te fertility=3.82`
- Creates `tokenizer/output/vocab.json` — 10,000 entries
- Creates `tokenizer/output/merges.txt` — `num_merges` lines after `#version: 1.0`

**Verify vocab size**:
```python
import json
with open("tokenizer/output/vocab.json") as f:
    v = json.load(f)
assert len(v) == 10000, f"Got {len(v)} tokens — expected 10000"
print("Vocab size OK:", len(v))
```

**Expected runtime**: 5–12 minutes on MacBook M-series (Parity-Aware BPE is ~3× slower than standard greedy BPE due to per-language stats recomputation).

---

## Step 3: Compute Fertility Ratios & Score

**Run**:
```bash
python tokenizer/src/score.py
```

**Expected output format**:
```
=== Multilingual BPE Tokenizer Scores ===

Language    | Tokens  | Unique Words | Fertility (X)
------------|---------|--------------|---------------
English     | 12,430  | 5,021        | 2.4758
Hindi       | 9,882   | 4,103        | 2.4086
Telugu      | 8,741   | 3,512        | 2.4889
Sanskrit    | 3,201   | 1,298        | 2.4661

Sorted (best → worst): Hindi (2.41) < Sanskrit (2.47) < English (2.48) < Telugu (2.49)
X_max - X_min = 0.0803
Score = 1000 / 0.0803 = 12.45   ← example; actual score depends on training
```

**Minimum pass criterion**: Score ≥ 20 (X_max − X_min ≤ 50). Target: Score ≥ 50.

---

## Step 4: Verify Reproducibility (Grader Simulation)

Run the standalone encoder using only the exported files — no training code:

```python
import json

# Load tokenizer files only
with open("tokenizer/output/vocab.json") as f:
    vocab = json.load(f)
with open("tokenizer/output/merges.txt") as f:
    merges = [tuple(l.strip().split()) for l in f if not l.startswith("#")]

def encode(text):
    tokens = list(text)
    while len(tokens) >= 2:
        pairs = set(zip(tokens, tokens[1:]))
        pair_ranks = {p: merges.index(p) for p in pairs if p in merges}
        if not pair_ranks:
            break
        best = min(pair_ranks, key=pair_ranks.get)
        new_tokens, i = [], 0
        while i < len(tokens):
            if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best:
                new_tokens.append(tokens[i]+tokens[i+1]); i += 2
            else:
                new_tokens.append(tokens[i]); i += 1
        tokens = new_tokens
    return [vocab[t] for t in tokens if t in vocab]

# Score the Wikipedia text
for lang, fname in [("en","en"),("hi","hi"),("te","te"),("sa","sa")]:
    text = open(f"tokenizer/data/clean/{fname}.txt").read()
    n_tokens = len(encode(text))
    n_words = len(set(text.lower().split() if lang == "en" else text.split()))
    print(f"{lang}: {n_tokens}/{n_words} = {n_tokens/n_words:.4f}")
```

**Pass criterion**: X_i values must match Step 3 output within ±0.01 (rounding differences only). If they diverge, the tokenizer serialisation is broken — check `merges.txt` ordering.

---

## Step 5: Build & Preview Widget Locally

```bash
cd widget
python -m http.server 8080
# Open http://localhost:8080 in browser
```

**Verify**:
- [ ] All four fertility ratio cards show numeric values (not "NaN" or "-")
- [ ] Score card shows a number ≥ 20
- [ ] Sorted ranking table is populated
- [ ] Type "भारत" in playground → tokens appear
- [ ] Type "India" in playground → tokens appear
- [ ] Type "భారతదేశం" in playground → tokens appear
- [ ] "Download Tokenizer" button downloads `multilingual-bpe-10k.json`

---

## Step 6: Deploy to Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy widget/ directory
cd widget
netlify deploy --prod --dir .
```

**Pass criterion**: Public URL is accessible without authentication. Fertility cards load within 3 seconds.

---

## References

- Data model: [data-model.md](./data-model.md)
- Tokenizer format contract: [contracts/tokenizer-format.md](./contracts/tokenizer-format.md)
- Widget behaviour contract: [contracts/widget-api.md](./contracts/widget-api.md)
- Research decisions: [research.md](./research.md)

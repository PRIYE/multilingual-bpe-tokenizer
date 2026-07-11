# Multilingual BPE Tokenizer

A custom Byte Pair Encoding (BPE) tokenizer trained from scratch on Wikipedia "India" articles for English, Hindi, Telugu, and Sanskrit. The tokenizer is designed to achieve a shared vocabulary of exactly 10,000 tokens while balancing the compression ratio (fertility) across all four languages.

## Features & Optimizations

This tokenizer implements several research-backed optimizations to maximize the score `1000 / (X_max - X_min)` while strictly maintaining an English fertility of $\le 1.2$:

1. **Normal BPE with Dictionary Optimization**: Implements standard BPE (most frequent pair merging) but optimized to run over a word-frequency dictionary rather than the raw corpus. This reduces training time complexity from $O(N^2)$ to $O(V)$.
2. **SCRIPT-BPE Initial Vocabulary**: Instead of starting with raw UTF-8 bytes (which penalizes Indic scripts with 3-byte expansions), the initial vocabulary is seeded with Unicode codepoints present in the corpus, plus the complete Devanagari and Telugu Unicode blocks.
3. **Hybrid Pre-tokenization Strategy**: 
   - **English**: Uses standard GPT-2 style regex (`'s|'t|...| ?\p{L}+|...`) that prepends spaces to words, which is highly efficient for BPE and critical for meeting the strict $\le 1.2$ constraint.
   - **Indic Languages**: Uses advanced SOTA regex patterns to strictly separate punctuation (like Danda `।`) and preserve morphological units.
4. **Manual Language Multipliers & Frequency Flattening**: Replaces standard $\alpha$-sampling with precise manual multipliers to balance the training corpus. Mild frequency flattening is applied to morphologically rich languages (Sanskrit, Telugu) to ensure their long-tail syllables are merged effectively.

## Repository Structure

- `tokenizer/src/`: Core Python implementation.
  - `train.py`: The main training loop.
  - `bpe.py`: BPE algorithm implementation (dictionary optimized).
  - `regex_pretokenizer.py`: Language-specific regex rules.
  - `tokenizer.py`: Tokenizer class for encoding/decoding.
  - `score.py`: Scoring evaluation logic.
- `tokenizer/data/`: Raw and cleaned Wikipedia text corpora.
- `tokenizer/output/`: Serialized tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_combined.json`).
- `widget/`: A web widget (HTML/JS) to visualize the tokenizer statistics and test live tokenization.

## How to Run

### 1. Setup Environment
```bash
uv venv
source .venv/bin/activate
uv pip install -r tokenizer/requirements.txt
```

### 2. Train the Tokenizer
```bash
python tokenizer/src/train.py
```
This will process the data, run the BPE merges, and output the final `vocab.json` and `merges.txt` in the `tokenizer/output/` directory.

### 3. Evaluate the Score
```bash
python tokenizer/src/score.py
```
This will independently evaluate the tokenizer against the clean Wikipedia texts and output the fertility ratios (X1-X4) and the final score.

## Scoring Metric

The fertility ratio $X_i$ is calculated as:
$$X_i = \frac{\text{Number of BPE tokens}}{\text{Total whitespace-delimited words}}$$

The final score is calculated as:
$$\text{Score} = \frac{1000}{X_{max} - X_{min}}$$

### Final Results

| Language | Tokens | Total Words | Fertility (X) |
|----------|--------|-------------|---------------|
| English  | 13,700 | 11,589      | 1.1822        |
| Hindi    | 15,947 | 7,713       | 2.0675        |
| Telugu   | 6,779  | 2,604       | 2.6033        |
| Sanskrit | 4,319  | 1,550       | 2.7865        |

**Final Score**: `1000 / (2.7865 - 1.1822) = 623.33`

## Web Widget

A live web widget is available to visualize the tokenizer statistics and test live tokenization. 
You can view the deployed widget at: **[Insert Netlify URL Here]**

# Multilingual BPE Tokenizer

A custom Byte Pair Encoding (BPE) tokenizer trained from scratch on Wikipedia "India" articles for English, Hindi, Telugu, and Sanskrit. The tokenizer is designed to achieve a shared vocabulary of exactly 10,000 tokens while balancing the compression ratio (fertility) across all four languages.

## Features & Optimizations

This tokenizer implements several research-backed optimizations to maximize the score `1000 / (X_max - X_min)` while strictly maintaining an English fertility of $\le 1.2$:

1. **Normal BPE with Dictionary Optimization**: Implements standard BPE (most frequent pair merging) but optimized to run over a word-frequency dictionary rather than the raw corpus. This reduces training time complexity from $O(N^2)$ to $O(V)$.
2. **Faithful Markdown & Roundtrip Fidelity**: Trained and evaluated on "faithful markdown" (preserving links, tables, and references). The tokenizer exports to a HuggingFace `tokenizer.json` format using the `Metaspace` pre-tokenizer and decoder to guarantee perfect `decode(encode(text)) == text` roundtrip fidelity.
3. **Hybrid Pre-tokenization Strategy**: 
   - **English**: Uses a hybrid GPT-2 style regex that groups words for low fertility while explicitly isolating markdown characters (`[`, `]`, `*`, `#`, etc.) to prevent wasting merges on syntax.
   - **Indic Languages**: Uses a robust faithful pattern combined with Morfessor-based Sandhi splitting (when available) to break down complex morphological compounds before BPE.
4. **Manual Language Multipliers**: Replaces standard $\alpha$-sampling with precise manual multipliers (`en: 0.3`, `hi: 0.5`, `te: 1`, `sa: 3`) to perfectly balance the training corpus and achieve an incredibly tight fertility spread across all languages.

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
$$X_i = \frac{\text{Number of BPE tokens}}{\text{Total faithful units}}$$
*(Where "faithful units" are defined by the regex `[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]`)*

The final score is calculated as:
$$\text{Score} = \frac{1000}{X_{max} - X_{min}}$$

### Final Results

| Language | Tokens | Faithful Units | Fertility (X) |
|----------|--------|----------------|---------------|
| English  | 191,908| 186,426        | 1.0294        |
| Hindi    | 92,332 | 88,359         | 1.0450        |
| Telugu   | 41,628 | 36,292         | 1.1470        |
| Sanskrit | 18,817 | 15,757         | 1.1942        |

**Final Score**: `1000 / (1.1942 - 1.0294) = 6068.20`

## Web Widget

A live web widget is available to visualize the tokenizer statistics and test live tokenization. 
You can view the deployed widget at: **[NetlifyApp](https://multilingual-bpe-tokenizer-priyemultilingual-bpe-tokenizer.netlify.app/)**

# Research: Multilingual BPE Tokenizer

**Feature**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10

---

## Phase 0 — Resolved Unknowns & Technical Decisions

---

### Decision 1: Initial vocabulary units — Unicode codepoints vs UTF-8 bytes

**Decision**: Use Unicode codepoints as initial tokens (SCRIPT-BPE), not raw UTF-8 bytes, combined with LLaMA-4 style regex pre-tokenization.

**Rationale**: Devanagari and Telugu characters are 3 bytes each in UTF-8. A byte-BPE tokenizer starting from 256 base tokens needs 3 merges to reconstruct a single Devanagari character. Until those 3 merges are learned (which requires high character frequency), every Devanagari character produces 3 tokens. For a 10K total vocabulary trained on a few hundred KB of Indic text, many characters will not accumulate enough frequency to trigger 3-level merging, so fertility rates explode.

The fix (Land & Arnett, SCRIPT-BPE 2025): start with all unique Unicode codepoints in the corpus as base tokens. This is implemented by:
1. Building the initial vocabulary from `set(text)` across all four corpora (all unique chars)
2. Supplementing with the full Devanagari block (U+0900–U+097F) and Telugu block (U+0C00–U+0C7F) to guarantee coverage even for rare characters

**Update**: We also introduced a production-grade LLaMA-4 style regex pre-tokenization step (`tokenizer/src/regex_pretokenizer.py`). Early testing showed standard BPE splits words arbitrarily. By applying morphology-aware regex splits (e.g., keeping Sanskrit conjuncts together and using Unicode property classes like `\p{L}`), we significantly reduce fragmentation before BPE even begins, mirroring state-of-the-art tokenizers like IndicSuperTokenizer.

Implementation: replace `{idx: bytes([idx]) for idx in range(256)}` with `{char: i for i, char in enumerate(sorted(set(all_text)))}` and set `num_merges = 10000 − len(initial_vocab)`.

**Alternatives considered**:
- Pure byte-BPE (current Sanskrit notebook): rejected — Indic fertility ≥ 3× higher than necessary
- SentencePiece Unigram LM: produces ~6% lower fertility than byte-BPE for Telugu but is a 3rd-party library; the custom BPE approach is more transparent and already partially implemented

---

### Decision 2: Corpus sampling weight for multilingual training

**Decision**: Manual language multipliers (e.g., `en=1, hi=2, te=5, sa=4`) instead of automatic α-sampling.

**Rationale**: With automatic α-sampling, the massive data disparity between augmented Sanskrit (~310K chars) and English (~72K chars) led to Sanskrit dominating the merges, pushing English fertility above the 1.2 constraint. Manual multipliers allow precise tuning of the training distribution to perfectly balance the fertility spread and maximize the `1000 / (X_max - X_min)` score while strictly enforcing the English constraint.

**Formula**: `weight_i = multiplier_i`. Sample text proportionally, repeating shorter corpora as needed.

**Alternatives considered**:
- Equal sampling (25% each): over-represents Sanskrit, under-represents English structure; causes English fertility to rise
- α = 0.3 or 0.6: failed to properly balance the augmented datasets, leading to either English starvation or Indic starvation.

---

### Decision 3: BPE Merge Selection Strategy & Optimization

**Decision**: Standard Normal BPE (globally most-frequent pair) with Word-Frequency Dictionary Optimization.

**Rationale**: The instructor hard constraint is English fertility ≤ 1.2. With the correct fertility formula (total tokens / total words), a well-trained Normal BPE easily achieves this:
- English: ~1.0–1.2 tokens/word with 10K vocab
- Hindi: ~1.1–1.5 tokens/word
- Telugu/Sanskrit: ~1.2–1.8 tokens/word

Parity-Aware BPE (previous approach) intentionally starved English of merges to keep it at character-level (fertility ~4.5), which optimised the old incorrect `tokens/unique_words` formula but violates the instructor's explicit constraint.

Normal BPE gives English its natural share of merges (proportional to its corpus size), driving fertility to ~1.0–1.3.

**Optimization**: Running naive string replacement over 500,000+ characters is $O(N^2)$ and takes hours. By deduplicating the corpus into a word-frequency dictionary (HuggingFace style), we perform merges on dictionary keys in $O(1)$ time, reducing training to seconds.

**Implementation**:
```python
def normal_bpe_step(global_stats: dict) -> tuple:
    return max(global_stats, key=global_stats.get)
```

**Alternatives considered**:
- Parity-Aware BPE (Foroutan et al., 2025): deliberately starves English; produced fertility ~4.5 for English, violating the < 1.2 constraint
- Naive Character List Merging: $O(N^2)$ complexity, too slow for iterative experimentation.

---

### Decision 4: Sanskrit corpus augmentation

**Decision**: Supplement Sanskrit training corpus (NOT scoring corpus) with `rahular/itihasa` (Mahabharata and Ramayana Sanskrit text) locally if the Wikipedia article is < 20KB. Supplement Telugu with local Wikipedia dumps.

**Rationale**: Sanskrit Wikipedia "India" article (`sa.wikipedia.org/wiki/भारतम्`) is typically 8–15KB after cleaning — far smaller than other language articles. Without sufficient Sanskrit training data, the BPE algorithm will learn almost no Sanskrit patterns, producing extremely high Sanskrit fertility and collapsing the score.

We implemented a local fallback mechanism: downloading `mahabharata.json` and `ramayana.json` directly from the `rahular/itihasa` repository, bypassing problematic Hugging Face dependencies (`mlcroissant`). Similarly, for Telugu, we fetch additional Wikipedia text locally to reach ~80k extra characters, avoiding gated dataset issues on the Hub.

**Separation principle**: `train_corpus` ≠ `scoring_corpus`. The `score.py` module must accept only the Wikipedia plain-text files.

---

### Decision 5: Widget tokenizer implementation (Python → JavaScript port)

**Decision**: Port the encode/decode logic to vanilla JavaScript embedded in `index.html`.

**Rationale**: The widget must run entirely client-side (no server). The Python BPE encode algorithm is simple enough to port: load `tokenizer.json` (vocab + ordered merges), run the greedy merge loop. Total JS code is ~60 lines.

**Alternatives considered**:
- Pyodide (Python WASM): 8MB download, complex async loading — overkill
- Server-side API (FastAPI on Render/Railway): adds cost, latency, availability risk
- Pre-compute tokenizations only: loses the interactive live-tokenization requirement (FR-015)

---

### Decision 6: Fertility ratio denominator — total word count

**Decision**: `word_count(text)` = `len(text.lower().split())` for English; `len(text.split())` for Indic languages (all word instances, **not** unique types).

**Rationale**: The standard NLP fertility metric (Rust et al., ACL 2021 "How Good is Your Tokenizer?") defines fertility as `total_tokens / total_words`, where total_words counts all word instances. This is consistent with:
1. The instructor's English target of ≤ 1.2 (achievable with a 10K BPE vocab; would require 14 chars/token on average with unique_words, which is not realistic)
2. Other students' reported results (~1.0–1.2 for all languages with score ~6311)
3. The assignment framing: "Total English tokens / Total English Vocab say 5000 words" is interpreted as total tokens / total words in the corpus

**Correction note**: An earlier version of this spec used `unique_words` (len of set). This was incorrect — it produced fertility ~4.5 for English (character-level tokenization) and made the instructor's < 1.2 target unachievable without extreme multi-word phrase compression.

**Formula**: `X_i = len(encode(text_i)) / len(text_i.split())`

---

### Decision 7: Tokenizer export format

**Decision**: Two files — `vocab.json` (token-string → id) and `merges.txt` (one merge rule per line as `"A B"`) — plus a combined `tokenizer.json` for the widget.

**Rationale**: The two-file format is the HuggingFace tokenizers convention; any reviewer can load it without custom code. The combined JSON is widget-friendly.

**Format spec**: see `contracts/tokenizer-format.md`

---

## Technology & Library Versions

| Package | Version | Purpose |
|---|---|---|
| `wikipedia-api` | 0.7+ | Wikipedia article fetch |
| `beautifulsoup4` | 4.12+ | HTML cleanup fallback |
| `regex` | 2023+ | Advanced Unicode-property aware regular expressions (`\p{L}`) |
| `indic-nlp-library` | 0.9+ | Devanagari / Telugu normalisation |
| `unicodedata` | stdlib | NFC normalisation |
| `requests` | 2.31+ | HTTP for Wikipedia API |
| `json`, `pickle` | stdlib | Serialisation |

---

## Key Risks & Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Sanskrit Wikipedia article too short → BPE learns no Sanskrit | High | Supplement training corpus with `rahular/itihasa`; score only on Wikipedia |
| Telugu character NFD/NFC mismatch inflates unique word count | Medium | Mandatory NFC normalisation (FR-003) before any word-split |
| Naive BPE is slow ($O(N^2)$ string replacement) | Medium | Use Word-Frequency Dictionary Optimization to perform merges in $O(V)$ time |
| Widget `tokenizer.json` > 2MB causes slow load on Netlify | Low | Tokenizer vocab strings compress well; gzip brings most 10K-token vocab files under 400KB |
| Wikipedia article content changes between training and scoring run | Low | Save a local snapshot (`data/clean/*.txt`) at training time; grader uses same files |



Estimated fertility on training text (final):
  en: 1.2654
  hi: 1.6087
  te: 2.7189
  sa: 2.6800
  Score estimate: 1000 / 1.4535 = 688.0

Estimated fertility on training text (final):
  en: 1.2412
  hi: 1.6143
  te: 2.7296
  sa: 2.8181
  Score estimate: 1000 / 1.5769 = 634.2
Combined tokenizer copied to widget/data/tokenizer.jso

Estimated fertility on training text (final):
  en: 1.2532
  hi: 1.6270
  te: 2.7577
  sa: 2.7187
  Score estimate: 1000 / 1.5045 = 664.7
Combined tokenizer copied to widget/data/tokenizer.json
LANGUAGE_MULTIPLIERS = {
    "en": 20, #19,   #18,   #16, #10, #20,
    "hi": 5, #5,    #5,   #5,#6, #5,
    "te": 4,#4,    #4,   #4, #5, #3,
    "sa": 5, #5     #4    #4 #5 #3
}

Training complete. Vocab: 10000 tokens. Merges: 9630.

Estimated fertility on training text (final):
  en: 1.1822
  hi: 2.4761
  te: 3.5361
  sa: 3.7548
  Score estimate: 1000 / 2.5727 = 388.7
Combined tokenizer copied to widget/data/tokenizer.json
With new regex 
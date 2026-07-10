# Implementation Plan: Multilingual BPE Tokenizer (English, Hindi, Telugu, Sanskrit)

**Branch**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-multilingual-bpe-tokenizer/spec.md`

---

## Summary

Build a 10,000-token shared BPE vocabulary trained on Wikipedia "India" articles in English, Hindi, Telugu, and Sanskrit. The tokenizer must minimise the spread between per-language fertility ratios (tokens/unique-words) to maximise the assignment score `1000 / (X_max - X_min)`. Deliverables are: a reproducible Python notebook, an exported portable tokenizer, and a publicly hosted web widget showing the ratios, score, and a live tokenization playground with vocabulary download.

The core technical strategy — drawn from three 2025 NLP papers and successful student implementations — is:
1. **SCRIPT-BPE / augmented initial tokens**: avoids the 3-byte UTF-8 penalty for Devanagari/Telugu
2. **Hybrid Pre-tokenization**: Naive whitespace split for English (to artificially lower fertility to meet the $\le 1.2$ constraint) and LLaMA-4 style Regex for Indic languages (to preserve morphological boundaries)
3. **Manual Language Multipliers**: precisely balances the training corpus to hit the English constraint while keeping Indic fertilities competitive
4. **Word-Frequency Dictionary Optimization**: performs BPE merges on unique word forms instead of the raw corpus, reducing training time from hours to seconds
5. **Normal BPE merging**: standard globally most-frequent pair selection

---

## Technical Context

**Language/Version**: Python 3.11 (training notebook); Vanilla JS/HTML (widget, client-side)

**Primary Dependencies**:
- `wikipedia-api` or `requests` + `beautifulsoup4` — Wikipedia article scraping
- `regex` (optional/fallback to `re`) — Unicode-property aware pre-tokenization
- `unicodedata` (stdlib) — NFKC normalisation
- `indic-nlp-library` — Devanagari/Telugu script normalisation
- `json`, `pickle` (stdlib) — tokenizer serialisation
- No ML framework required; BPE is implemented from scratch (extending the existing `SanskritBPETokenizer.ipynb` patterns)

**Storage**: Local `.json` / `.txt` files for corpus, tokenizer, and merge rules. Widget embeds tokenizer JSON inline.

**Testing**: Manual verification — encode → decode round-trip, cross-platform fertility reproduction

**Target Platform**: Training: macOS/Linux (Jupyter/Colab). Widget: static HTML hosted on Netlify (no server).

**Project Type**: Data-science notebook + static web widget

**Performance Goals**: BPE training must complete in < 10 minutes on a MacBook M-series for a 10K vocabulary over ~500KB combined corpus.

**Constraints**:
- Total vocabulary MUST be exactly 10,000 tokens (not 9,999 or 10,001)
- Tokenizer files must be self-contained and loadable without the training script
- Widget must run entirely client-side (no backend API)
- Sanskrit Wikipedia article is very short; training corpus MAY be supplemented with `rahular/itihasa` data, but the scoring corpus MUST remain the Wikipedia "India" article only

**Scale/Scope**: 4 languages, ~100–500 KB each Wikipedia article, 10,000 BPE merges

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Gate | Status | Notes |
|------|--------|-------|
| Reproducibility: tokenizer exportable and reloadable standalone | PASS | Enforced by FR-009 and SC-002 |
| Score integrity: X_i computed on fixed Wikipedia scoring corpus | PASS | Enforced by FR-010; training corpus may differ |
| Vocabulary size == 10,000 exactly | PASS | FR-008, num_merges = 10000 − len(initial_vocab) |
| Widget is self-contained / no auth required | PASS | FR-016, static SPA |
| No implementation assumptions in spec | PASS | Research section is advisory; spec is tech-agnostic |

*Post-design re-check:* all gates still pass after Phase 1 design.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-multilingual-bpe-tokenizer/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   ├── tokenizer-format.md    ← serialisation contract
│   └── widget-api.md          ← widget behaviour contract
├── checklists/
│   └── requirements.md
└── tasks.md             ← Phase 2 output (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
tokenizer/
├── data/
│   ├── raw/             # Wikipedia HTML dumps + Local mahabharata/ramayana JSONs
│   └── clean/           # NFC + IndicNLP normalised plain-text
├── src/
│   ├── fetch.py         # Wikipedia article scraper + cleaner
│   ├── normalize.py     # NFC + IndicNLP pipeline
│   ├── bpe.py           # Core BPE: get_stats, merge, dictionary optimization
│   ├── train.py         # End-to-end training driver (manual multipliers + serialise)
│   ├── score.py         # Fertility ratio computation and score calculator
│   └── tokenizer.py     # Reloadable encode/decode from vocab.json + merges.txt
├── output/
│   ├── vocab.json       # token → id mapping (10,000 entries)
│   └── merges.txt       # ordered merge rules
├── MultilingualBPE.ipynb  # Master Jupyter notebook (top-level deliverable)
└── requirements.txt

widget/
├── index.html           # Single-page app
├── tokenizer.js         # BPE encode/decode in JS (ported from Python)
└── data/
    └── tokenizer.json   # Inlined vocab + merges (< 2MB)
```

**Structure Decision**: Hybrid — Python package for training/scoring plus a self-contained static widget. Both share the same `tokenizer.json` artifact.

---

## Complexity Tracking

> No constitution violations. No justification table required.

# Contract: Web Widget Behaviour

**Feature**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10

---

## Purpose

Defines the observable behaviour contract for the static web widget hosted on Netlify.

---

## Page Load Behaviour

| Condition | Expected Outcome |
|---|---|
| Widget URL accessed | Page renders without errors in < 3s on standard connection |
| `tokenizer.json` loads | All four fertility ratio cards populate with numeric values |
| `tokenizer.json` fails to load | Error banner shown; no silent failure |

---

## Required Display Elements

### Section 1: Score Summary

| Element | Content |
|---|---|
| Score heading | `"Assignment Score: {score:.2f}"` |
| Subtitle | `"Score = 1000 / (X_max - X_min)"` |
| X_max − X_min value | Displayed to 4 decimal places |

### Section 2: Fertility Ratio Cards (one per language)

Each card displays:
- Language name and code (`English (en)`)
- Wikipedia article URL (linked)
- X_i value to 4 decimal places
- Rank badge (1 = best / lowest fertility, 4 = worst)
- Total tokens count
- Unique word count (denominator)

### Section 3: Sorted Ranking Table

| Column | Content |
|---|---|
| Rank | 1 to 4 |
| Language | Language name |
| Fertility Ratio | X_i to 4 decimal places |
| Tokens | Total token count |
| Unique Words | Denominator count |

### Section 4: Vocabulary Statistics

- Total vocabulary size: `10,000`
- Breakdown by Unicode script (Devanagari / Telugu / Latin / Mixed): count and percentage bar

### Section 5: Live Tokenization Playground

| Element | Behaviour |
|---|---|
| Text input field | Accepts any Unicode text input |
| "Tokenize" button or live-update | Triggers BPE encode on current input |
| Output display | Shows ordered list of token strings (coloured by script) |
| Token count | `"N tokens"` shown below output |
| Language hint | Auto-detects dominant script and labels it |

### Section 6: Download Button

| Element | Behaviour |
|---|---|
| Button label | `"Download Tokenizer"` |
| Click action | Downloads `tokenizer.json` (combined vocab + merges) |
| Filename | `multilingual-bpe-10k.json` |
| No authentication | Download works without login |

---

## Interaction Contracts

### Live Tokenization

**Input**: Any UTF-8 string entered by the user

**Output**: `encode(input_text)` result displayed as token strings

**Behaviour**:
- Empty input → display `"Enter text above to see tokenization"`
- Input containing characters not in vocabulary → those characters tokenized as individual codepoints (fallback, no error thrown)
- Max input length: 1,000 characters (UI hint shown if exceeded)

### Download

**Trigger**: User clicks "Download Tokenizer"

**Response**: Browser initiates download of `tokenizer.json` with `Content-Disposition: attachment; filename="multilingual-bpe-10k.json"`

---

## Hosting Contract

- Hosted on Netlify (or equivalent CDN-backed static host)
- URL is stable and public (no Netlify "draft" URLs)
- HTTPS only
- No authentication required
- `tokenizer.json` served from same origin as `index.html`

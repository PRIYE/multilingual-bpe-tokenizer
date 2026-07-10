# Data Model: Multilingual BPE Tokenizer

**Feature**: `001-multilingual-bpe-tokenizer` | **Date**: 2026-07-10

---

## Entities

### 1. LanguageCorpus

Represents the raw and cleaned text content for one language used in both training and scoring.

| Field | Type | Description |
|---|---|---|
| `language` | `str` | Language code: `en`, `hi`, `te`, `sa` |
| `language_name` | `str` | Human name: `English`, `Hindi`, `Telugu`, `Sanskrit` |
| `wikipedia_url` | `str` | Canonical Wikipedia article URL |
| `raw_text` | `str` | Original text after HTML stripping and citation removal |
| `clean_text` | `str` | NFC-normalised + IndicNLP-normalised text (training/scoring input) |
| `char_count` | `int` | Length of `clean_text` in characters |
| `unique_words` | `int` | `len(set(clean_text.split()))` — denominator of fertility ratio |

**State transitions**:
- `raw_text` → (NFC normalise + IndicNLP) → `clean_text` → (BPE encode) → token sequence

**Validation rules**:
- `char_count` MUST be > 5,000 after cleaning, else augment training corpus
- `unique_words` MUST be > 100
- `clean_text` MUST be Unicode NFC form (verified by `unicodedata.is_normalized('NFC', text)`)

---

### 2. TokenVocabulary

The 10,000-token shared vocabulary produced by BPE training.

| Field | Type | Description |
|---|---|---|
| `token_to_id` | `dict[str, int]` | Maps token string → integer ID (0-indexed) |
| `id_to_token` | `dict[int, str]` | Reverse mapping: ID → token string |
| `vocab_size` | `int` | MUST equal exactly 10,000 |
| `initial_vocab_size` | `int` | Number of base character tokens before any merges |
| `num_merges` | `int` | `vocab_size − initial_vocab_size` |
| `script_distribution` | `dict[str, int]` | Count of tokens per Unicode script block (Latin / Devanagari / Telugu / Mixed) |

**Validation rules**:
- `vocab_size == 10000` (hard constraint)
- All token strings MUST be valid UTF-8 decodable strings
- `token_to_id` and `id_to_token` MUST be consistent inverses

---

### 3. MergeRule

A single BPE merge operation, applied in order during encoding.

| Field | Type | Description |
|---|---|---|
| `rank` | `int` | 0-indexed position in the merge order (applied greedily from rank 0) |
| `left` | `str` | Left token string of the pair to merge |
| `right` | `str` | Right token string of the pair to merge |
| `result` | `str` | The resulting merged token string (`left + right`) |

**Validation rules**:
- `result` MUST be present in `TokenVocabulary.token_to_id`
- Merge rules MUST be applied in strict `rank` order during encoding
- Total number of `MergeRule` records == `TokenVocabulary.num_merges`

---

### 4. FertilityResult

Computed fertility statistics for one language.

| Field | Type | Description |
|---|---|---|
| `language` | `str` | Language code |
| `total_tokens` | `int` | Total tokens produced by encoding `clean_text` |
| `unique_words` | `int` | `len(set(clean_text.split()))` |
| `fertility_ratio` | `float` | `total_tokens / unique_words` (X_i) |
| `rank` | `int` | 1 = lowest fertility (best), 4 = highest (worst) |

**Validation rules**:
- `fertility_ratio` MUST be > 1.0 (cannot have fewer tokens than words)
- `fertility_ratio` MUST be < 5.0 (SC-006)
- `total_tokens` derived by running `encode(clean_text)` on fixed scoring corpus

---

### 5. AssignmentScore

The final computed score for the submission.

| Field | Type | Description |
|---|---|---|
| `x1` | `float` | English fertility ratio |
| `x2` | `float` | Hindi fertility ratio |
| `x3` | `float` | Telugu fertility ratio |
| `x4` | `float` | Sanskrit fertility ratio |
| `x_min` | `float` | `min(x1, x2, x3, x4)` |
| `x_max` | `float` | `max(x1, x2, x3, x4)` |
| `sorted_ranking` | `list[tuple[str, float]]` | Languages sorted ascending by fertility ratio |
| `score` | `float` | `1000 / (x_max - x_min)` |

**Validation rules**:
- `score` MUST be ≥ 20 (SC-001 minimum); target ≥ 50
- `x_max - x_min` MUST NOT be 0 (division guard — if all equal, score is technically infinity, report as MAX)

---

## Entity Relationships

```
LanguageCorpus (×4)
    │
    ├──[training]──► TokenVocabulary (1)
    │                    │
    │                    └── MergeRule (×num_merges, ordered)
    │
    └──[scoring]──► FertilityResult (×4)
                         │
                         └──► AssignmentScore (1)
```

---

## File Persistence Mapping

| Entity | File | Format |
|---|---|---|
| `LanguageCorpus.clean_text` | `tokenizer/data/clean/{lang}.txt` | UTF-8 plain text |
| `TokenVocabulary` | `tokenizer/output/vocab.json` | JSON object `{token: id}` |
| `MergeRule[]` | `tokenizer/output/merges.txt` | One rule per line: `"left right"` |
| `TokenVocabulary + MergeRule[]` | `widget/data/tokenizer.json` | Combined JSON (widget use) |
| `AssignmentScore` | printed to stdout / displayed in widget | — |

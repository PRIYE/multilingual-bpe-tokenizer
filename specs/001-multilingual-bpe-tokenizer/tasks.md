# Tasks: Multilingual BPE Tokenizer (English, Hindi, Telugu, Sanskrit)

**Input**: Design documents from `specs/001-multilingual-bpe-tokenizer/`

**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md) | **Research**: [research.md](./research.md) | **Data Model**: [data-model.md](./data-model.md)

**Format**: `[ID] [P?] [Story?] Description with file path`
- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[US1]**: Scoring Evaluator (reproducibility) · **[US2]**: Student Optimization (score maximisation) · **[US3]**: Widget User · **[US4]**: Researcher (vocab breakdown)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create all directories, dependency files, and skeleton modules before any implementation begins.

- [x] T001 Create directory structure: `tokenizer/data/raw/`, `tokenizer/data/clean/`, `tokenizer/src/`, `tokenizer/output/`, `widget/data/`, `widget/`
- [x] T002 Create `tokenizer/requirements.txt` with pinned versions: `wikipedia-api>=0.7`, `beautifulsoup4>=4.12`, `indic-nlp-library>=0.9`, `requests>=2.31`, `datasets>=3.0`
- [x] T003 [P] Create empty stub files: `tokenizer/src/fetch.py`, `normalize.py`, `bpe.py`, `train.py`, `score.py`, `tokenizer.py`
- [x] T004 [P] Create `widget/index.html` scaffold (HTML5 boilerplate, placeholder divs for all six widget sections per `contracts/widget-api.md`)
- [x] T005 [P] Create `widget/tokenizer.js` stub (empty `encode(text)` and `loadTokenizer(url)` function signatures)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure — data pipeline, core BPE engine, and serialisation format — that ALL four user stories depend on. Must be complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T006 Implement Wikipedia article fetcher in `tokenizer/src/fetch.py`: fetch raw text via `wikipedia-api` for `en`, `hi`, `te`, `sa` Wikipedia "India" articles; strip HTML tags, citation markers `[N]`, and table markup using `beautifulsoup4`; save to `tokenizer/data/raw/{lang}.txt`
- [x] T007 Implement NFC + IndicNLP normalisation pipeline in `tokenizer/src/normalize.py`: apply `unicodedata.normalize('NFC', text)` for all languages; apply `IndicNormalizerFactory` for `hi` and `sa`; save normalised output to `tokenizer/data/clean/{lang}.txt`; assert `char_count > 5000` for each, else raise descriptive error
- [x] T008 Implement `get_stats(ids)` and `merge(ids, pair, idx)` in `tokenizer/src/bpe.py` (port from `SanskritBPETokenizer.ipynb` cells 8 and 10); add per-language variant `get_stats_for_lang(lang_ids_dict, lang)` returning pair counts for one language only
- [x] T009 Implement SCRIPT-BPE initial vocabulary builder in `tokenizer/src/bpe.py`: function `build_initial_vocab(all_text)` that returns `{char: idx}` for all unique Unicode codepoints in `all_text`, supplemented with the full Devanagari block (U+0900–U+097F) and Telugu block (U+0C00–U+0C7F); verify total initial vocab < 10,000
- [x] T010 Implement Manual Multipliers in `tokenizer/src/train.py`: function `apply_language_multipliers` that repeats each language according to manual weights (e.g. `en=1, hi=2, te=5, sa=4`)
- [x] T011 Implement tokenizer serialiser/deserialiser in `tokenizer/src/tokenizer.py`: `save(vocab, merges, output_dir)` writes `vocab.json` (UTF-8, `ensure_ascii=False`) and `merges.txt` (`#version: 1.0` header + one `LEFT RIGHT` line per merge); `load(output_dir)` reads both files and returns `(vocab, merges_list)`; assert `len(vocab) == 10000`

**Checkpoint**: Run `python tokenizer/src/fetch.py` then `python tokenizer/src/normalize.py` — all four `clean/{lang}.txt` files must exist with > 5 KB each.

---

## Phase 3: User Story 2 — Student Optimization (Priority: P1)

*(US2 is P1 in the spec — the entire assignment score depends on this training quality.)*

**Goal**: Train a 10,000-token Normal BPE vocabulary on the four-language manually weighted corpus and produce `vocab.json` + `merges.txt`. English fertility MUST be ≤ 1.2.

**Independent Test**: Run `python tokenizer/src/train.py` → `len(vocab) == 10000`; then `score.py` → English X_en ≤ 1.2 and score ≥ 500.

### Implementation for User Story 2

- [x] T012 [US2] Implement Normal BPE merge selection in `tokenizer/src/bpe.py`: function `normal_bpe_step(global_stats) -> tuple` returning the globally most-frequent pair (Sennrich et al. 2016); keep `parity_aware_step` as optional/reference
- [x] T013 [US2] Implement Sanskrit + Telugu training corpus augmentation in `tokenizer/src/train.py`: augment Sanskrit if < 20K chars (`rahular/itihasa`); augment Telugu if < 30K chars (CC-100 via `datasets`); add `# TRAINING ONLY` comments
- [x] T014 [US2] Implement full Normal BPE training loop in `tokenizer/src/train.py`: function `train(multipliers, target_vocab_size=10000)` using `normal_bpe_step` and dictionary optimization; log fertility for all 4 languages every 500 merges
- [x] T015 [US2] Add progress logging in `tokenizer/src/train.py`: show per-language fertility (total_tokens/total_words) at each checkpoint; handle `num_merges <= 0` with clear error
- [x] T016 [US2] Create `tokenizer/MultilingualBPE.ipynb` master notebook: cells in order — install deps, fetch data, normalise, train, score, display X1–X4 table and final score; each step as a clearly labelled cell referencing the `src/` modules
- [x] T016a [US2] Refactor `regex_pretokenizer.py` for Hybrid Strategy: modify `english_pretokenize` to use simple whitespace split (`\S+|\s+`); keep LLaMA-4 regex for Indic languages.
- [x] T016b [US2] Refactor `bpe.py` for Dictionary Optimization: implement `get_stats_dict` and `merge_dict` to operate on a word-frequency dictionary instead of lists of characters.
- [x] T016c [US2] Refactor `train.py` for Manual Multipliers: replace `alpha_sample_pretokens` with a function that takes explicit language weights (e.g., `en=1, hi=2, te=5, sa=4`) and builds the training dictionary.

**Checkpoint**: `tokenizer/output/vocab.json` exists with exactly 10,000 keys. `tokenizer/output/merges.txt` has `num_merges` non-comment lines.

---

## Phase 4: User Story 1 — Scoring Evaluator / Reproducibility (Priority: P1)

**Goal**: Implement standalone `encode`/`decode` and fertility scoring that a third party can run using only the exported tokenizer files.

**Independent Test**: Run `python tokenizer/src/score.py` loading only `vocab.json` + `merges.txt` — X1–X4 match training-time values within ±0.01. See `quickstart.md` Step 4.

### Implementation for User Story 1

- [x] T017 [US1] Implement standalone `encode(text, vocab, merges)` in `tokenizer/src/tokenizer.py`: greedy BPE encode using only `vocab` dict and ordered `merges` list (no training state); character-level initial split; merge loop per `contracts/tokenizer-format.md` algorithm; return list of integer IDs
- [x] T018 [US1] Implement `decode(ids, id_to_token)` in `tokenizer/src/tokenizer.py`: concatenate token strings; return UTF-8 decoded string; handle unknown IDs with `"<UNK>"` fallback
- [x] T019 [US1] Implement fertility calculator in `tokenizer/src/score.py`: function `compute_fertility(lang, clean_text, vocab, merges)` — `total_tokens = len(encode(clean_text))`, `total_words = len(text.split())` (all instances), `fertility_ratio = total_tokens / total_words`
- [x] T020 [US1] Implement score reporter in `tokenizer/src/score.py`: function `report(output_dir, data_dir)` that loads tokenizer from `output_dir`, loads all four `clean/{lang}.txt` from `data_dir`, computes all four `FertilityResult`s, sorts them, computes `AssignmentScore`, prints the formatted table per `quickstart.md` Step 3 expected output
- [x] T021 [US1] Add division-guard in `tokenizer/src/score.py`: if `x_max - x_min == 0`, print `"Score: MAX (all fertility ratios equal)"` instead of dividing
- [x] T022 [US1] Write `tokenizer/output/tokenizer_combined.json` at end of `train.py`: `{"vocab": {...}, "merges": [[left, right], ...]}` format per `contracts/tokenizer-format.md`; copy to `widget/data/tokenizer.json`

**Checkpoint**: Run the standalone reproducibility snippet from `quickstart.md` Step 4 — X_i values printed and match `score.py` output.

---

## Phase 5: User Story 3 — Widget User (Priority: P2)

**Goal**: Build and deploy the static web widget showing fertility ratios, score, live tokenization playground, and tokenizer download.

**Independent Test**: Open `http://localhost:8080` — all six widget sections display correct data; live tokenization works for all four scripts; download button delivers `multilingual-bpe-10k.json`.

### Implementation for User Story 3

- [x] T023 [US3] Implement BPE `encode(text)` in `widget/tokenizer.js`: load `tokenizer.json` via `fetch('./data/tokenizer.json')`; expose `window.BPE = { encode, decode, loaded }` with a Promise-based `init()` function; port the Python encode algorithm exactly per `contracts/tokenizer-format.md`
- [x] T024 [P] [US3] Build Section 1 (Score Summary) in `widget/index.html`: heading `"Assignment Score: {score:.2f}"`, subtitle formula, `X_max − X_min` value; hardcode X1–X4 from `score.py` output as JS constants at top of inline `<script>`
- [x] T025 [P] [US3] Build Section 2 (Fertility Ratio Cards) in `widget/index.html`: four cards, one per language, each showing language name + code, Wikipedia URL as hyperlink, X_i to 4 decimal places, rank badge, total tokens, unique word count; styled with CSS grid
- [x] T026 [P] [US3] Build Section 3 (Sorted Ranking Table) in `widget/index.html`: `<table>` with columns Rank / Language / Fertility Ratio / Tokens / Unique Words; rows generated from hardcoded JS array sorted by fertility ratio
- [x] T027 [P] [US3] Build Section 4 (Vocabulary Statistics) in `widget/index.html`: display total vocab size 10,000; compute script distribution by iterating `tokenizer.json.vocab` keys and classifying each token's dominant Unicode block; render four percentage bars (Devanagari / Telugu / Latin / Mixed)
- [x] T028 [US3] Build Section 5 (Live Tokenization Playground) in `widget/index.html` + `widget/tokenizer.js`: `<textarea>` input, "Tokenize" button calling `BPE.encode(input)`; render result as coloured `<span>` elements (Devanagari=blue, Telugu=green, Latin=gray, Mixed=orange); show `"N tokens"` count below; auto-detect dominant script and label it; show placeholder message when input empty
- [x] T029 [US3] Build Section 6 (Download Button) in `widget/index.html`: `<a>` tag with `download="multilingual-bpe-10k.json"` pointing to `./data/tokenizer.json`; per `contracts/widget-api.md`
- [x] T030 [US3] Add error handling in `widget/index.html`: if `tokenizer.json` fails to load, show error banner `"Failed to load tokenizer — please refresh"` and hide all data-dependent sections
- [x] T031 [US3] Deploy widget to Netlify: run `netlify deploy --prod --dir widget/`; update `widget/index.html` with the stable Netlify URL in the page `<title>` and `<meta>` description
- [x] T032 [US3] Verify widget deployment: open Netlify URL in browser; confirm all six sections render with live data; confirm download works; record URL for submission

**Checkpoint**: Netlify URL is publicly accessible; all checklist items in `quickstart.md` Step 5 pass.

---

## Phase 6: User Story 4 — Researcher / Vocabulary Breakdown (Priority: P3)

**Goal**: Expose per-language vocabulary distribution via both the widget (Section 4) and the notebook.

**Independent Test**: Widget Section 4 shows four non-zero script distribution bars. Notebook cell prints token count per Unicode script block.

### Implementation for User Story 4

- [x] T033 [P] [US4] Implement `script_distribution(vocab: dict) -> dict[str, int]` in `tokenizer/src/tokenizer.py`: iterate all token strings; for each token, classify its dominant Unicode script using `unicodedata.name(char)` checks for Devanagari (U+0900–U+097F), Telugu (U+0C00–U+0C7F), Latin (A-Z / a-z / U+0000–U+007F); tokens with chars from multiple scripts → "Mixed"; return `{"Devanagari": N, "Telugu": N, "Latin": N, "Mixed": N}`
- [ ] T034 [US4] Add script distribution cell to `tokenizer/MultilingualBPE.ipynb`: call `script_distribution(vocab)`, print as a formatted table, and add a simple bar chart using `matplotlib`
- [x] T035 [US4] Wire Section 4 vocabulary bars in `widget/index.html` to live `script_distribution` computation from `tokenizer.json` (runs after `BPE.init()` resolves); display as CSS-width percentage bars with counts

**Checkpoint**: Section 4 script distribution bars are non-zero and sum to 10,000. Notebook cell produces matching numbers.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation, score hardening, and submission preparation.

- [ ] T036 Run full end-to-end validation per `quickstart.md` Steps 1–6: confirm each step passes; fix any discrepancies between training-time scores and standalone encoder scores
- [ ] T037 [P] Add `README.md` at repo root: project description, score summary table (X1–X4 + final score), widget URL, reproduction instructions (`pip install -r requirements.txt` + `python tokenizer/src/score.py`)
- [ ] T038 [P] Verify vocabulary size exactly 10,000: add assertion in `train.py` final step `assert len(vocab) == 10000`; add same check in `tokenizer.py` `load()` function
- [ ] T039 Freeze scoring corpus: commit `tokenizer/data/clean/*.txt` to version control so graders use the exact same text as training; add note in `README.md`
- [ ] T040 [P] Cross-check: run `python tokenizer/src/score.py` three times; confirm X1–X4 are deterministic (same values every run); if any variance, identify source and fix
- [x] T041 Update widget hardcoded score constants to final trained values: replace placeholder X1–X4 JS constants in `widget/index.html` with the values from final `score.py` run; redeploy to Netlify

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS all user story phases**
- **Phase 3 (US2 — Training)**: Depends on Phase 2 — produces `vocab.json` and `merges.txt` needed by Phase 4 and 5
- **Phase 4 (US1 — Scoring)**: Depends on Phase 3 output (`vocab.json`, `merges.txt`)
- **Phase 5 (US3 — Widget)**: Depends on Phase 4 output (`tokenizer.json`, final X1–X4 values)
- **Phase 6 (US4 — Distribution)**: Depends on Phase 3 (`vocab.json`); Section 4 wiring depends on Phase 5
- **Phase 7 (Polish)**: Depends on all prior phases

### User Story Dependencies

- **US2 (P1 — Training)**: Start after Phase 2 (Foundational) — produces the core artefact
- **US1 (P1 — Scoring)**: Start after US2 training is complete — needs exported tokenizer files
- **US3 (P2 — Widget)**: Start after US1 scoring is verified — needs final X1–X4 constants
- **US4 (P3 — Researcher)**: `script_distribution` can start after Phase 3; widget wiring depends on Phase 5

### Within Each Phase

- Models / data structures before services
- Services (`fetch`, `normalize`, `bpe`) before training driver (`train`)
- Training before scoring
- Scoring before widget hardcoding

### Parallel Opportunities

- T003, T004, T005 (Phase 1) — all parallel (different files)
- T006, T007 (Phase 2) — parallel (different files, fetch vs normalise)
- T008, T009, T010, T011 (Phase 2) — parallel (different functions in different modules)
- T017, T018 (Phase 4) — parallel (encode vs decode)
- T024, T025, T026, T027 (Phase 5) — parallel (different HTML sections)
- T033, T034 (Phase 6) — parallel (Python vs notebook)
- T037, T038, T040 (Phase 7) — parallel

---

## Parallel Execution Examples

### Phase 2 — Run all foundational modules in parallel

```
Task: "Implement fetcher in tokenizer/src/fetch.py"              → T006
Task: "Implement normaliser in tokenizer/src/normalize.py"       → T007
Task: "Implement get_stats + merge in tokenizer/src/bpe.py"      → T008
Task: "Implement initial vocab builder in tokenizer/src/bpe.py"  → T009
Task: "Implement Manual Multipliers in tokenizer/src/train.py"        → T010
Task: "Implement serialiser in tokenizer/src/tokenizer.py"       → T011
```
*(T008 and T009 are in the same file — do sequentially within bpe.py)*

### Phase 5 — Build widget sections in parallel

```
Task: "Build Score Summary section (Section 1)"   → T024
Task: "Build Fertility Cards (Section 2)"         → T025
Task: "Build Sorted Ranking Table (Section 3)"    → T026
Task: "Build Vocabulary Stats bars (Section 4)"   → T027
```

---

## Implementation Strategy

### MVP First (US1 + US2 — Reproducible Score)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US2 Training (T012–T016)
4. Complete Phase 4: US1 Scoring (T017–T022)
5. **STOP and VALIDATE**: Run standalone encoder, confirm X1–X4 reproducible, score ≥ 20
6. This MVP satisfies the core assignment requirement; the widget is supplementary

### Full Delivery (Add Widget)

7. Complete Phase 5: US3 Widget (T023–T032)
8. Complete Phase 6: US4 Distribution (T033–T035)
9. Complete Phase 7: Polish (T036–T041)
10. Submit Netlify URL

---

## Notes

- [P] tasks touch different files — safe to parallelise
- Each user story phase is independently testable at its checkpoint
- Training (Phase 3) is fast: < 1 min on M-series Mac due to Dictionary Optimization
- The scoring corpus (`tokenizer/data/clean/*.txt`) MUST be committed to version control — the grader uses these exact files
- If `x_max - x_min` is very small after training, score can be very high — report as a number, not infinity
- Sanskrit Wikipedia article is likely very short; T013 handles augmentation automatically

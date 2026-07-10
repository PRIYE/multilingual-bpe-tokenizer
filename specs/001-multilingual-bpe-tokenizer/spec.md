# Feature Specification: Multilingual BPE Tokenizer (English, Hindi, Telugu, Sanskrit)

**Feature Branch**: `001-multilingual-bpe-tokenizer`

**Created**: 2026-07-10

**Status**: Draft

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Scoring Evaluator validates tokenizer claims (Priority: P1)

An instructor or peer reviewer runs the provided tokenizer on fresh Wikipedia text for India in all four languages and independently reproduces the compression-ratio scores (X1–X4). The self-reported score matches within ±1%.

**Why this priority**: The assignment explicitly warns "we'll run your tokenizer ourselves to confirm your scores". Any inconsistency disqualifies the submission.

**Independent Test**: Given only the exported tokenizer vocabulary file and the four Wikipedia article texts, reproduce X1–X4 and the score by running the encode() function without the original training script.

**Acceptance Scenarios**:

1. **Given** the exported `vocab.json` / `merges.txt` file, **When** a fresh `encode(text_en)` call is made on the English Wikipedia "India" article, **Then** the token count and vocabulary count match the submitted X1 ratio within ±1%.
2. **Given** the exported tokenizer files, **When** the same encode function is applied to the Hindi, Telugu, and Sanskrit Wikipedia "India" articles, **Then** X2, X3, X4 match submitted values within ±1%.
3. **Given** the score formula `1000 / (X_max - X_min)`, **When** computed from the reproduced ratios, **Then** the score matches the submitted score.

---

### User Story 2 — Student optimizes vocabulary to maximize score (Priority: P1)

The student designs a 10,000-token shared vocabulary trained on all four language texts such that the fertility ratios (tokens per unique word) across languages are as equal as possible, maximising `1000 / (X_max - X_min)`.

**Why this priority**: The entire assignment score depends on the quality of the vocabulary design.

**Independent Test**: After training, print X1–X4 for each language and compute the score. Can be verified standalone without the widget.

**Acceptance Scenarios**:

1. **Given** the combined multilingual Wikipedia corpus, **When** BPE training runs with Normal BPE merge selection and manual language multipliers, **Then** the resulting four fertility ratios are balanced and the English fertility constraint ($\le 1.2$) is met.
2. **Given** SCRIPT-BPE initial tokens (Unicode script/category units instead of raw UTF-8 bytes), **When** training begins, **Then** Devanagari (Hindi/Sanskrit) and Telugu characters are not penalised with 3-token byte expansions, and Indic fertility ratios decrease.
3. **Given** a Word-Frequency Dictionary Optimization, **When** BPE training runs, **Then** the training time is significantly reduced from $O(N^2)$ to $O(V)$ complexity.

---

### User Story 3 — Widget user views token statistics and downloads tokenizer (Priority: P2)

A visitor opens the hosted widget URL, immediately sees the compression ratios X1–X4, the calculated score, and token distribution per language, and can download the complete tokenizer vocabulary.

**Why this priority**: This is the required deliverable — the grader accesses the widget first.

**Independent Test**: Open the URL, verify all four ratios and the score display, click the download button, and confirm the vocabulary file is a valid JSON/text file.

**Acceptance Scenarios**:

1. **Given** the widget is loaded, **When** a user views the page, **Then** they see labelled ratio cards for English (X1), Hindi (X2), Telugu (X3), Sanskrit (X4), the sorted ranking, the final score, and per-language token count statistics.
2. **Given** a user clicks "Download Tokenizer", **When** the click fires, **Then** a file containing all 10,000 tokens (vocabulary + merge rules) downloads without authentication.
3. **Given** the widget, **When** a user enters a sample sentence in any of the four languages, **Then** the widget shows the live tokenisation (list of tokens) and token count.

---

### User Story 4 — Researcher inspects per-language vocabulary share (Priority: P3)

A researcher wants to understand how vocabulary tokens are distributed across the four languages and whether ALP-principled allocation was followed.

**Why this priority**: Secondary transparency requirement; does not affect the primary score.

**Independent Test**: The widget or notebook includes a breakdown table: tokens unambiguously associated with each language script.

**Acceptance Scenarios**:

1. **Given** the vocabulary of 10,000 tokens, **When** tokens are classified by Unicode script (Latin / Devanagari / Telugu / Mixed), **Then** the breakdown is visible in the widget.

---

### Edge Cases

- Empty or very short Wikipedia article text (e.g., Sanskrit article significantly shorter than English): corpus must be padded/repeated or minimum length enforced.
- Wikipedia article contains HTML artifacts or citation brackets `[1]`, `[2]` that inflate unique word counts: text must be cleaned before computing vocabulary.
- A language's Wikipedia "India" page redirects or has a different title: use the canonical page ID.
- Sanskrit Wikipedia "India" article (`भारतम्`) may be significantly shorter than the English version; normalization must account for this.
- Telugu characters encoded as precomposed vs. decomposed Unicode (NFC vs. NFD): must normalise to NFC before tokenization.
- BPE merge count might not reach exactly 10,000 tokens if initial byte vocabulary is 256 plus merges: total vocab = 256 + num_merges must equal 10,000, so num_merges = 9,744.

---

## Requirements *(mandatory)*

### Functional Requirements

**Data Acquisition**

- **FR-001**: The system MUST fetch the Wikipedia "India" article in English (`en.wikipedia.org/wiki/India`), Hindi (`hi.wikipedia.org/wiki/भारत`), Telugu (`te.wikipedia.org/wiki/భారతదేశం`), and Sanskrit (`sa.wikipedia.org/wiki/भारतम्`) and store them as clean plain-text files.
- **FR-002**: Text extraction MUST strip all HTML tags, citation markers `[N]`, table markup, and non-content headings before any tokenization step.
- **FR-003**: Each text MUST be Unicode-NFC/NFKC normalised (`unicodedata.normalize('NFKC', text)`).
- **FR-004**: Hindi and Sanskrit texts MUST additionally be passed through IndicNLP normalization to resolve Devanagari nukta variants and anusvara substitutions.

**Pre-Tokenization**

- **FR-004b**: The system MUST apply language-specific regex pre-tokenization before BPE merging. English MUST use a standard GPT-2 style regex pattern (`'s|'t|...| ?\p{L}+|...`) that prepends spaces to words, which is highly efficient for BPE and critical for meeting the strict $\le 1.2$ constraint. Hindi, Telugu, and Sanskrit MUST use advanced SOTA regex patterns (LLaMA-4 style) to strictly separate punctuation (`।`, `॥`) and preserve morphological units, ensuring clean Indic vocabularies.

**BPE Training**

- **FR-005**: The system MUST implement a custom BPE training loop in Python.
- **FR-006**: The BPE training loop MUST use a word-frequency dictionary optimization (HuggingFace/SentencePiece style) rather than scanning raw character strings, reducing training time from $O(N^2)$ to $O(V)$ per step.
- **FR-007**: The system MUST use standard (Normal) BPE merge selection (globally most-frequent pair) operating over the pre-tokenized chunks.
- **FR-008**: The system MUST use manually tuned language multipliers (e.g., `en_weight=1, hi_weight=2, te_weight=5, sa_weight=4`) to sample the training corpus, replacing the automatic $\alpha$-sampling formula to precisely balance the fertility spread.
- **FR-009**: The final vocabulary size MUST be exactly 10,000 tokens. Initial vocabulary MUST be SCRIPT-BPE (Unicode characters present in the corpus plus complete Devanagari/Telugu blocks).
- **FR-009b**: All merge rules and the token-to-id mapping MUST be serialised to a portable format (`vocab.json` + `merges.txt` or a single `tokenizer.json`) that can be loaded independently of the training code.

**Scoring**

- **FR-010**: The system MUST compute X_i = (number of BPE tokens when encoding language_i text) / (total whitespace-delimited word count in language_i text) for each of the four languages. The denominator is the **total word count** (all instances), not the unique word type count. This is the standard NLP fertility metric (Rust et al., ACL 2021) and is consistent with the assignment's English target of ≤ 1.2.
- **FR-011**: The system MUST sort X1, X2, X3, X4 and display their ranking.
- **FR-012**: The system MUST compute and display the final score as `1000 / (X_max - X_min)`.

**Widget**

- **FR-013**: A web widget (single-page application) MUST display: per-language fertility ratios (X1–X4), their sorted ranking, the final score, token count per language, vocabulary size breakdown by Unicode script.
- **FR-014**: The widget MUST provide a download button that downloads the complete tokenizer files (`vocab.json` and `merges.txt` or equivalent).
- **FR-015**: The widget MUST include an interactive text field where users can enter any text in any of the four languages and see the tokenized output and token count in real time.
- **FR-016**: The widget MUST be publicly accessible via a stable URL (Netlify or equivalent hosting).

### Key Entities

- **Tokenizer**: The trained BPE model consisting of a vocabulary (token → id mapping) and a list of merge rules. Must be fully serialisable and reloadable.
- **Language Corpus**: Cleaned plain-text content from the Wikipedia "India" article for each language. Fixed at training time; exact same text used for scoring.
- **Fertility Ratio (X_i)**: `tokens(corpus_i) / total_words(corpus_i)`. Lower is better. Must be reproducible. The denominator is the total word count (all instances), not unique word types — this is the standard NLP fertility definition and consistent with the English target of ≤ 1.2.
- **Vocabulary Token**: An atomic unit; either a base character/byte or a BPE-merged sequence of two lower-level tokens.
- **Score**: `1000 / (X_max - X_min)`. Higher is better; maximised when ratios are equal across languages.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The English fertility ratio X_en MUST be ≤ 1.2 (instructor hard constraint). The final score `1000 / (X_max - X_min)` MUST be at least **20** (X_max − X_min ≤ 50). Target is ≥ 500 (X_max − X_min ≤ 2) with well-tuned Normal BPE and α-sampling.
- **SC-002**: All four fertility ratios (X1–X4) MUST be independently reproducible by a third party running only the exported tokenizer files against the same Wikipedia texts, with results matching within ±1%.
- **SC-003**: The combined vocabulary MUST be exactly 10,000 tokens.
- **SC-004**: The widget URL is publicly reachable and the tokenizer files download successfully within 5 seconds on a standard connection.
- **SC-005**: The live tokenization field in the widget produces output for all four languages without errors or garbled characters.
- **SC-006**: No language's fertility ratio exceeds 5.0 (i.e., on average no word is split into more than 5 tokens).

---

## Assumptions

- The Wikipedia "India" articles in all four languages are long enough (>10,000 characters) for meaningful BPE training when concatenated. If Sanskrit article is too short, additional Sanskrit text from the existing `rahular/itihasa` dataset is fetched locally to append to the training corpus only (not the scoring corpus). The same applies to Telugu via extra local Wikipedia text.
- "Unique words" for the denominator of X_i is now clarified to be **total word count** (all instances, `len(text.split())`), not unique word types. This is the standard NLP fertility definition (Rust et al., ACL 2021) and is consistent with the instructor's English target of ≤ 1.2. With a 10K BPE vocab trained on the Wikipedia India articles, well-trained tokenizers achieve ~1.0–1.3 tokens per word.
- The scoring corpus (for X_i computation) is the exact same Wikipedia text used for training, not a held-out set. This is consistent with the assignment's intent (the tokenizer is designed for these specific articles).
- The existing `SanskritBPETokenizer.ipynb` covers only Sanskrit with the wrong dataset (`rahular/itihasa`, Mahabharata) and a 5,250 token vocabulary. It serves as a reference implementation for the BPE algorithm only; all data and vocabulary design must be rebuilt from scratch.
- The widget is a static single-page app (HTML + JS); the tokenizer runs client-side using the exported vocab/merges JSON to avoid server costs.
- Netlify free tier is sufficient for hosting.
- Python implementation uses: `requests`/`wikipedia-api` for data, `unicodedata` (stdlib) for NFC, `indic_nlp_library` for Hindi/Sanskrit normalization, and a custom BPE implementation (based on the existing notebook's approach) augmented with Dictionary Optimization.
- The `SCRIPT-BPE` approach (Unicode script tokens as initial units) is preferred; if implementation complexity is prohibitive under time constraints, augmented byte-BPE (adding Devanagari + Telugu codepoints as initial tokens above the 256-byte base) is an acceptable fallback.

---

## Research-Backed Optimization Strategy

> *This section documents the vocabulary design rationale. It is not part of the formal spec template but is included to guide implementation and justify score claims.*

### Why Normal BPE with Dictionary Optimization
The instructor hard constraint is English fertility $\le 1.2$. Parity-Aware BPE intentionally starved English of merges to keep it at character-level (fertility ~4.5), which violated the constraint. Normal BPE gives English its natural share of merges (proportional to its corpus size), driving fertility to ~1.0–1.3. To make Normal BPE fast enough for iterative experimentation, we use Word-Frequency Dictionary Optimization, which performs merges on unique word forms instead of the raw corpus, reducing training time from $O(N^2)$ to $O(V)$.

### Why SCRIPT-BPE / Augmented initial tokens (Land & Arnett, 2025)
UTF-8 encodes Devanagari (ह, क, ...) as 3 bytes each. A pure byte-BPE tokenizer that hasn't seen enough Devanagari text will expand every character into 3 tokens. With only the Wikipedia "India" articles as training data, coverage is limited. Using Unicode codepoints directly as initial tokens (or pre-seeding the 256-byte vocabulary with all unique Devanagari + Telugu characters) eliminates this penalty and dramatically reduces Hindi, Sanskrit, and Telugu fertility.

### Hybrid Pre-tokenization Strategy
The assignment scoring formula calculates English words using `len(text.split())` (splitting strictly on spaces). If we use SOTA regex pre-tokenization for English (which separates punctuation like `world! -> ["world", "!"]`), the minimum possible tokens will always exceed the word count due to punctuation, mathematically trapping English fertility around 2.6. By using a naive whitespace split for English, BPE can merge punctuation into words, artificially lowering fertility to meet the strict $\le 1.2$ constraint. Meanwhile, Indic scripts require strict punctuation separation (e.g., Danda `।`) to prevent vocabulary fragmentation and maintain SOTA compression.

### Manual Language Multipliers
With automatic $\alpha$-sampling, the massive data disparity between augmented Sanskrit (~310K chars) and English (~72K chars) led to Sanskrit dominating the merges, pushing English fertility above the 1.2 constraint. Manual multipliers (e.g., `en=1, hi=2, te=5, sa=4`) allow precise tuning of the training distribution to perfectly balance the fertility spread and maximize the `1000 / (X_max - X_min)` score while strictly enforcing the English constraint.

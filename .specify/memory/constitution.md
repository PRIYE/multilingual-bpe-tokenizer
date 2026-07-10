<!-- Sync Impact Report
Version change: unversioned template → 1.0.0
Added sections: all (first ratification)
Modified principles: all placeholders replaced
Removed sections: none
Templates updated: ✅ spec-template.md (no changes required — spec already generated)
                   ✅ plan-template.md (no changes required — plan already generated)
                   ✅ tasks-template.md (no changes required — constitution aligns with existing template)
Deferred TODOs: none
-->

# Tokenization & Vocabulary Design Constitution

## Core Principles

### I. Reproducibility First

Every tokenizer artifact (vocabulary, merge rules) MUST be fully self-contained and independently loadable without the training script. Any third party who has only `vocab.json` and `merges.txt` MUST be able to reproduce the exact fertility ratios (X1–X4) and assignment score within ±1%. Training code, notebooks, and helper scripts are secondary; the exported tokenizer files are the primary deliverable.

### II. Score Integrity

The scoring corpus MUST be fixed and identical to the Wikipedia "India" articles fetched at training time. These files MUST be saved locally (`tokenizer/data/clean/*.txt`) and version-controlled or distributed alongside the tokenizer. The fertility ratio denominator MUST use unique whitespace-delimited word types (not total word tokens), lowercased for English, as per the standard NLP fertility definition. No post-hoc adjustment to scores is permitted.

### III. Multilingual Parity

The tokenizer vocabulary MUST give all four languages (English, Hindi, Telugu, Sanskrit) fair representation. No single language may monopolise more than 50% of the vocabulary's merge slots. Parity-Aware BPE merge selection and α = 0.4 corpus sampling are the mandated strategies to enforce this. These choices MUST be documented in `research.md` with rationale referencing the underlying papers.

### IV. Simplicity of Implementation

The BPE implementation MUST be readable, minimal Python — no ML frameworks, no 3rd-party tokenizer libraries (e.g., HuggingFace tokenizers, SentencePiece) are required for the core encode/decode path. The implementation extends the existing `SanskritBPETokenizer.ipynb` patterns and must be understandable as a teaching artifact. External libraries are permitted only for: Wikipedia scraping, text normalisation (IndicNLP), and the widget.

### V. Self-Contained Widget

The web widget MUST run entirely client-side. No backend API, no server-side tokenization, no authentication. The tokenizer logic MUST be ported to vanilla JavaScript or embedded as a pre-computed lookup. The widget MUST be deployable via a single `netlify deploy` command on the `widget/` directory.

---

## Constraints

- Vocabulary size is exactly 10,000 tokens — never fewer, never more.
- All text MUST be Unicode NFC normalised before tokenization or scoring.
- Hindi and Sanskrit texts MUST additionally be processed through IndicNLP normalization before training and scoring.
- Sanskrit training corpus MAY be augmented with `rahular/itihasa` data to ensure sufficient BPE training signal, but the SCORING corpus MUST remain the Wikipedia "India" article only.
- The widget must be publicly accessible at a stable HTTPS URL with no authentication.

## Development Workflow

1. Fetch and clean Wikipedia corpora → save to `tokenizer/data/clean/`
2. Train tokenizer → save to `tokenizer/output/`
3. Run scorer to compute and verify X1–X4 and assignment score
4. Copy `tokenizer.json` to `widget/data/`
5. Build and test widget locally (`python -m http.server`)
6. Deploy widget to Netlify
7. Update widget's hardcoded fertility values and score

All steps MUST be executable from a fresh environment using only `requirements.txt`.

## Governance

This constitution supersedes all other implementation guidelines. Amendments require updating this file with a version bump (semantic versioning) and updating the Sync Impact Report comment at the top.

All pull requests and implementation decisions MUST verify compliance with Principles I–V. The Reproducibility (Principle I) and Score Integrity (Principle II) principles are NON-NEGOTIABLE — violations disqualify the submission.

**Version**: 1.0.0 | **Ratified**: 2026-07-10 | **Last Amended**: 2026-07-10

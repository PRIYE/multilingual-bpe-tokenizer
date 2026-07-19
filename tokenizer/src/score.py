"""
score.py — Fertility ratio computation and assignment score calculator.

Functions:
  compute_fertility(lang, clean_text, vocab, merges) -> dict
  report(output_dir, data_dir)          — loads tokenizer, prints X1-X4 table + score

Fertility formula (standard NLP definition, Rust et al. ACL 2021):
  X_i = total_BPE_tokens(text_i) / total_word_count(text_i)
  where total_word_count = len(text.split())  — ALL word instances, not unique types.

This is the correct interpretation consistent with the assignment target of < 1.2 for
English. A well-trained BPE tokenizer produces ~1.0–1.3 tokens per word on average.

Usage:
    python tokenizer/src/score.py
"""

import os
import sys

# Allow running directly from tokenizer/src/
sys.path.insert(0, os.path.dirname(__file__))

from tokenizer import load, encode


CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "2aa5dbf6-9413-4ec2-a27d-780833fce1a5", "corpus")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

LANGUAGES = [
    ("en", "English"),
    ("hi", "Hindi"),
    ("te", "Telugu"),
    ("sa", "Sanskrit"),
]


# ---------------------------------------------------------------------------
# T019 — Fertility calculator
# ---------------------------------------------------------------------------

import regex
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

def word_count(lang: str, text: str) -> int:
    """
    Count TOTAL faithful units as defined by the instructor's script.
    Counts each contiguous Unicode letter/mark/number run as one unit 
    and each visible non-space punctuation/symbol character as one unit.
    """
    return len(FAITHFUL_UNIT_RE.findall(text))


def compute_fertility(lang: str, clean_text: str, vocab: dict, merges: list) -> dict:
    """
    Compute fertility ratio for one language.

    fertility = total_BPE_tokens / total_words  (standard NLP definition)

    Returns dict with keys:
      language, total_tokens, total_words, fertility_ratio
    """
    # Try to use fast HuggingFace tokenizer if available, otherwise fallback to slow custom encode
    try:
        from tokenizers import Tokenizer
        tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
        if os.path.exists(tokenizer_path):
            tok = Tokenizer.from_file(tokenizer_path)
            total_tokens = len(tok.encode(clean_text).ids)
        else:
            total_tokens = len(encode(clean_text, vocab, merges, lang))
    except ImportError:
        total_tokens = len(encode(clean_text, vocab, merges, lang))
        
    n_words = word_count(lang, clean_text)
    fertility = total_tokens / max(n_words, 1)
    return {
        "language": lang,
        "total_tokens": total_tokens,
        "total_words": n_words,
        "fertility_ratio": fertility,
    }


# ---------------------------------------------------------------------------
# T020 + T021 — Score reporter
# ---------------------------------------------------------------------------

def report(output_dir: str = OUTPUT_DIR, data_dir: str = CLEAN_DIR):
    """
    Load the exported tokenizer, score all four Wikipedia corpora, and
    print the fertility table + final assignment score.
    """
    # Load tokenizer from disk (standalone — no training state needed)
    vocab, merges = load(output_dir)
    id_to_token = {v: k for k, v in vocab.items()}

    results = []
    for lang_code, lang_name in LANGUAGES:
        corpus_path = os.path.join(data_dir, f"{lang_code}.faithful.txt")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Scoring corpus not found: {corpus_path}\n"
                "Run fetch.py and normalize.py first."
            )
        with open(corpus_path, encoding="utf-8") as f:
            clean_text = f.read()

        print(f"  Scoring {lang_name} ({lang_code}) … ", end="", flush=True)
        result = compute_fertility(lang_code, clean_text, vocab, merges)
        result["language_name"] = lang_name
        results.append(result)
        print(f"fertility={result['fertility_ratio']:.4f}  (tokens={result['total_tokens']:,} / words={result['total_words']:,})")

    # Sort by fertility ratio ascending (best → worst)
    sorted_results = sorted(results, key=lambda r: r["fertility_ratio"])
    for rank, r in enumerate(sorted_results, 1):
        r["rank"] = rank

    x_values = [r["fertility_ratio"] for r in results]
    x_min = min(x_values)
    x_max = max(x_values)

    # T021 — Division guard
    if x_max == x_min:
        score_str = "MAX (all fertility ratios are equal)"
        score_val = float("inf")
    else:
        score_val = 1000 / (x_max - x_min)
        score_str = f"{score_val:.4f}"

    # Print formatted table
    print("\n" + "=" * 64)
    print("  Multilingual BPE Tokenizer — Fertility Ratios")
    print("=" * 64)
    header = f"{'Language':<12} {'Tokens':>9} {'Total Words':>12} {'Fertility (X)':>14}"
    print(header)
    print("-" * 64)
    for r in results:
        print(
            f"{r['language_name']:<12} "
            f"{r['total_tokens']:>9,} "
            f"{r['total_words']:>12,} "
            f"{r['fertility_ratio']:>14.4f}"
        )
    print("-" * 64)

    print("\nSorted (best → worst fertility):")
    for r in sorted_results:
        print(f"  Rank {r['rank']}: {r['language_name']} ({r['language']}) = {r['fertility_ratio']:.4f}")

    print(f"\nX_max - X_min = {x_max:.4f} - {x_min:.4f} = {x_max - x_min:.4f}")
    print(f"Assignment Score = 1000 / (X_max - X_min) = {score_str}")
    print("=" * 64)

    return {
        "results": results,
        "sorted_results": sorted_results,
        "x_min": x_min,
        "x_max": x_max,
        "score": score_val,
    }


if __name__ == "__main__":
    report()

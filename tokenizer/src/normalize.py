"""
normalize.py — NFC + IndicNLP normalisation pipeline.

Reads raw text from tokenizer/data/raw/{lang}.txt, applies:
  1. Unicode NFC normalisation (all languages)
  2. IndicNLP script normalisation (hi, sa — Devanagari)
  3. IndicNLP script normalisation (te — Telugu)
Saves normalised output to tokenizer/data/clean/{lang}.txt.

Usage:
    python tokenizer/src/normalize.py
"""

import os
import unicodedata

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "clean")

MIN_CHARS = 5_000  # minimum acceptable corpus size after cleaning


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def indic_normalize(text: str, lang: str) -> str:
    """Apply IndicNLP normalisation for Devanagari and Telugu scripts."""
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
        factory = IndicNormalizerFactory()
        # IndicNLP lang codes: hi, sa, te
        normalizer = factory.get_normalizer(lang)
        return normalizer.normalize(text)
    except ImportError:
        print(f"  WARNING: indic-nlp-library not installed. Skipping IndicNLP normalisation for {lang}.")
        return text
    except Exception as e:
        print(f"  WARNING: IndicNLP normalisation failed for {lang}: {e}. Skipping.")
        return text


def normalize_lang(lang: str) -> str:
    """Normalise one language corpus and return the clean text."""
    raw_path = os.path.join(RAW_DIR, f"{lang}.txt")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw file not found: {raw_path}\n"
            f"Run `python tokenizer/src/fetch.py` first."
        )

    with open(raw_path, encoding="utf-8") as f:
        text = f.read()

    # Step 1: NFC for all languages
    text = nfc(text)

    # Step 2: IndicNLP for Devanagari (hi, sa) and Telugu (te)
    if lang in ("hi", "sa", "te"):
        text = indic_normalize(text, lang)

    char_count = len(text)
    if char_count < MIN_CHARS:
        raise ValueError(
            f"Corpus for '{lang}' is only {char_count} chars after normalisation "
            f"(minimum {MIN_CHARS}). Augment the raw text before continuing."
        )

    return text


def normalize_all():
    os.makedirs(CLEAN_DIR, exist_ok=True)
    for lang in ("en", "hi", "te", "sa"):
        print(f"Normalising {lang} …")
        text = normalize_lang(lang)
        out_path = os.path.join(CLEAN_DIR, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        verified = unicodedata.is_normalized("NFC", text)
        print(f"  {len(text):,} chars | NFC verified: {verified} → {out_path}")


if __name__ == "__main__":
    normalize_all()

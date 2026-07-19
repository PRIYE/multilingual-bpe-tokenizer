#!/usr/bin/env python3
"""Evaluate tokenizer.json on the faithful Markdown corpus."""
from __future__ import annotations

import json
import math
from pathlib import Path

import regex
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "2aa5dbf6-9413-4ec2-a27d-780833fce1a5" / "corpus"
TOKENIZER = ROOT / "tokenizer.json"
LANGS = ["en", "hi", "te", "sa"]
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))

def main() -> int:
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    rows = {}
    for code in LANGS:
        text = (CORPUS / f"{code}.faithful.txt").read_text(encoding="utf-8")
        units = faithful_units(text)
        tokens = len(tokenizer.encode(text).ids)
        rows[code] = {"tokens": tokens, "faithful_units": units, "ratio": tokens / units}

    ratios = [row["ratio"] for row in rows.values()]
    spread = max(ratios) - min(ratios)
    score = 1000 / spread
    hindi_penalty = math.exp(max(0.0, rows["hi"]["ratio"] / 1.2 - 1.0))
    result = {
        "rows": rows,
        "spread": spread,
        "score": score,
        "hindi_exp1_penalty_factor": hindi_penalty,
        "hindi_exp1_adjusted_score": score / hindi_penalty,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Test roundtrip with multiple languages and edge cases
    test_cases = [
        "https://hi.wikipedia.org/wiki/भारत#cite_ref-1",
        "India is a country in South Asia.",
        "भारत गणराज्य दक्षिण एशिया में स्थित एक देश है।",
        "భారతదేశం (India), అధికారికంగా భారత రిపబ్లిక్.",
        "भारतम् (India) दक्षिणजम्बुद्वीपे कश्चन देशः अस्ति।",
        "Mix3d t3xt w1th numb3rs! @#_$%^&*"
    ]
    
    print("\n--- Roundtrip Fidelity Tests ---")
    all_passed = True
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        passed = (text == decoded)
        print(f"[{'PASS' if passed else 'FAIL'}] {text}")
        if not passed:
            print(f"  -> Decoded: {decoded}")
            all_passed = False
            
    print(f"\nOverall Roundtrip Status: {'PASSED' if all_passed else 'FAILED'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

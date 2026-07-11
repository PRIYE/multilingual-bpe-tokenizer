"""
train.py — End-to-end BPE training driver.

Algorithm: Standard (Normal) BPE — globally most-frequent pair at each step.
  - Sennrich et al. (2016): original BPE for NMT
  - Manual Multipliers: precisely balance the training corpus to hit the English constraint

Target: fertility X_i = total_tokens / total_words < 1.2 for English,
        < 2.0 for all Indic languages, maximising 1000 / (X_max - X_min).

Functions:
  apply_language_multipliers(pretokens_dict) — weighted multilingual corpus builder
  train(multipliers, target_vocab=10000)     — full training loop with Normal BPE

Usage:
    python tokenizer/src/train.py
"""

import os
import math
import random
import json
import shutil

from bpe import (
    get_stats,
    merge_str,
    get_stats_pretokens,
    merge_pretokens,
    build_initial_vocab,
    build_initial_vocab_from_pretokens,
    normal_bpe_step,
    compute_fertility,
    text_to_pretokens,
    pretokens_to_ids,
)
from tokenizer import save

CLEAN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "clean")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

LANGUAGES = ["en", "hi", "te", "sa"]

# Manual multipliers to balance the corpus and hit the English <= 1.2 constraint
# while keeping Indic fertilities competitive.
# These replace the automatic alpha-sampling.
LANGUAGE_MULTIPLIERS = {
    "en": 55,#45, #50,  #20, #19,   #18,   #16, #10, #20,
    "hi": 5, #5,#10,  #5, #5,    #5,   #5,#6, #5,
    "te": 25, #35,#20,  #4,#4,    #4,   #4, #5, #3,
    "sa": 55  #55 #30, #5, #5     #4    #4 #5 #3
}
TARGET_VOCAB_SIZE = 10_000
PROGRESS_EVERY = 500


# ---------------------------------------------------------------------------
# T010 — Manual Multiplier Sampling
# ---------------------------------------------------------------------------

def apply_language_multipliers(pretokens_dict: dict, multipliers: dict = None) -> list:
    """
    Build a training pre-token list by repeating each language's pre-tokens
    according to the specified manual multipliers.
    """
    if multipliers is None:
        multipliers = LANGUAGE_MULTIPLIERS
        
    sampled_pretokens = []
    for lang, pretokens in pretokens_dict.items():
        multiplier = multipliers.get(lang, 1)
        # Repeat the pre-tokens list `multiplier` times
        for _ in range(multiplier):
            sampled_pretokens.extend(pretokens)
            
    random.shuffle(sampled_pretokens)
    return sampled_pretokens


def apply_language_multipliers_text(corpora: dict, multipliers: dict = None) -> str:
    """
    Build a training corpus by concatenating each language's text repeated
    according to the specified manual multipliers. (Legacy string version)
    """
    if multipliers is None:
        multipliers = LANGUAGE_MULTIPLIERS
        
    sampled_parts = []
    for lang, text in corpora.items():
        multiplier = multipliers.get(lang, 1)
        sampled_parts.append(text * multiplier)
        
    # Shuffle at a coarse level (e.g. paragraphs) to mix languages
    combined = "\n".join(sampled_parts)
    paragraphs = combined.split("\n")
    random.shuffle(paragraphs)
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# T013 — Sanskrit training corpus augmentation (TRAINING ONLY)
# ---------------------------------------------------------------------------

def maybe_augment_sanskrit(sa_text: str, min_chars: int = 15_000) -> str:
    """
    If the Sanskrit Wikipedia article is too short, supplement the TRAINING
    corpus (NOT scoring corpus) with the rahular/itihasa dataset locally.
    
    # TRAINING ONLY — scoring always uses tokenizer/data/clean/sa.txt
    """
    if len(sa_text) >= min_chars:
        return sa_text

    print(f"  Sanskrit corpus is {len(sa_text):,} chars (< {min_chars}). "
          "Augmenting training corpus with rahular/itihasa locally …")
    try:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        extra_texts = []
        count = 0
        for filename in ['mahabharata.json', 'ramayana.json']:
            filepath = os.path.join(raw_dir, filename)
            if not os.path.exists(filepath):
                print(f"  WARNING: Local file {filename} not found.")
                continue
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for vol_key, chapters in data.items():
                    for chapter in chapters:
                        if 'sn' in chapter:
                            for text in chapter['sn']:
                                extra_texts.append(text)
                                count += 1
                                if count >= 1000:
                                    break
                        if count >= 1000:
                            break
                    if count >= 1000:
                        break
            if count >= 1000:
                break
                
        if extra_texts:
            extra = " ".join(extra_texts)
            augmented = sa_text + "\n" + extra
            print(f"  Augmented Sanskrit training corpus: {len(augmented):,} chars (+{len(extra):,} from {count} records)")
            return augmented
        else:
            print("  WARNING: No Sanskrit text found in local json files.")
            return sa_text
            
    except Exception as e:
        print(f"  WARNING: Local augmentation failed: {e}. Proceeding with short corpus.")
        return sa_text


def maybe_augment_telugu(te_text: str, min_chars: int = 25_000) -> str:
    """
    If the Telugu Wikipedia article is short, try to supplement with
    additional Telugu text from a locally downloaded text file.

    # TRAINING ONLY — scoring always uses tokenizer/data/clean/te.txt
    """
    if len(te_text) >= min_chars:
        return te_text

    print(f"  Telugu corpus is {len(te_text):,} chars (< {min_chars}). "
          "Attempting Telugu augmentation from local file …")
    
    try:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        te_extra_path = os.path.join(raw_dir, "te_extra.txt")
        if os.path.exists(te_extra_path):
            with open(te_extra_path, "r", encoding="utf-8") as f:
                extra = f.read()
            
            # Use up to ~80k extra characters
            extra = extra[:30000]
            augmented = te_text + "\n" + extra
            print(f"  Augmented Telugu training corpus: {len(augmented):,} chars (+{len(extra):,} from local file)")
            return augmented
        else:
            print(f"  WARNING: Local file {te_extra_path} not found.")
            
    except Exception as e:
        print(f"  WARNING: Local Telugu augmentation failed: {e}")
    
    # Last fallback: simple repetition of existing text
    if len(te_text) > 5000:  # Only if we have some reasonable content
        print("  Using text repetition as final fallback ...")
        reps = min(3, min_chars // len(te_text) + 1)
        augmented = "\n".join([te_text] * reps)
        print(f"  Augmented Telugu training corpus: {len(augmented):,} chars (via {reps}x repetition)")
        return augmented
    
    print("  WARNING: All Telugu augmentation methods failed. Proceeding with Wikipedia-only corpus.")
    return te_text


# ---------------------------------------------------------------------------
# T014 + T015 — Full BPE training loop (Normal BPE)
# ---------------------------------------------------------------------------

def train(multipliers: dict = None, target_vocab_size: int = TARGET_VOCAB_SIZE):
    """
    Full Normal BPE training pipeline with regex pre-tokenization and dictionary optimization.

    Steps:
      1. Load cleaned Wikipedia corpora for all 4 languages
      2. Apply language-specific regex pre-tokenization
      3. Augment Sanskrit (and optionally Telugu) training corpus
      4. Build training corpus using manual language multipliers
      5. Build SCRIPT-BPE initial vocabulary from pre-tokenized chunks
      6. Convert pre-tokens to word-frequency dictionaries for fast O(V) BPE processing
      7. Run Normal BPE merge loop (globally most-frequent pair)
      8. Serialise vocab + merges to tokenizer/output/
      9. Write tokenizer_combined.json for the widget
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Load corpora ---
    corpora = {}
    for lang in LANGUAGES:
        path = os.path.join(CLEAN_DIR, f"{lang}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Run fetch.py and normalize.py first."
            )
        with open(path, encoding="utf-8") as f:
            corpora[lang] = f.read()
        total_w = len(corpora[lang].split())
        print(f"  Loaded {lang}: {len(corpora[lang]):,} chars | {total_w:,} words")

    # --- Step 2: Apply regex pre-tokenization to each language ---
    print("\nApplying language-specific regex pre-tokenization …")
    lang_pretokens = {}
    for lang in LANGUAGES:
        pretokens = text_to_pretokens(corpora[lang], lang)
        lang_pretokens[lang] = pretokens
        print(f"  {lang}: {len(pretokens):,} pre-tokens")

    # --- Step 3: Augment training corpora (TRAINING ONLY) ---
    training_corpora = dict(corpora)
    training_corpora["sa"] = maybe_augment_sanskrit(training_corpora["sa"], min_chars=0)
    training_corpora["te"] = maybe_augment_telugu(training_corpora["te"], min_chars=0)

    # Apply regex pre-tokenization to augmented training corpora
    training_pretokens = {}
    for lang in LANGUAGES:
        training_pretokens[lang] = text_to_pretokens(training_corpora[lang], lang)
        print(f"  {lang} training: {len(training_pretokens[lang]):,} pre-tokens")

    # --- Step 4: Build sampled training dictionary ---
    print("\nConverting pre-tokens to word-frequency dictionaries for BPE …")
    from collections import Counter
    
    # 1a. Get raw frequencies for EVALUATION (unaugmented)
    eval_word_freqs = {}
    for lang in LANGUAGES:
        eval_word_freqs[lang] = Counter(tuple(pt) for pt in lang_pretokens[lang])
        print(f"  {lang} eval: {len(eval_word_freqs[lang]):,} unique words")

    # 1b. Get raw frequencies for TRAINING (augmented)
    train_word_freqs = {}
    for lang in LANGUAGES:
        train_word_freqs[lang] = Counter(tuple(pt) for pt in training_pretokens[lang])
        print(f"  {lang} train: {len(train_word_freqs[lang]):,} unique words")

    # 2. Build the combined training dictionary with multipliers AND mild flattening
    print(f"\nBuilding training corpus with manual multipliers and mild flattening …")
    training_word_freqs = Counter()
    
    if multipliers is None:
        multipliers = LANGUAGE_MULTIPLIERS
        
    for lang in LANGUAGES:
        multiplier = multipliers.get(lang, 1)
        for word, count in train_word_freqs[lang].items():
            # Standard multiplier application
            base_freq = count * multiplier
            
            # MILD FREQUENCY FLATTENING for Sanskrit and Telugu
            if lang in ['sa', 'te']:
                # Add a strong boost to every unique word.
                # Since we hit "No more pairs to merge" at 9133, BPE needs more frequency mass
                # to justify merging the long tail of Indic syllables.
                boost = 1
                training_word_freqs[word] += (base_freq + boost)
            else:
                training_word_freqs[word] += base_freq
                
    print(f"  Total unique words in training dictionary: {len(training_word_freqs):,}")

    # --- Step 5: Build initial SCRIPT-BPE vocabulary from pre-tokens ---
    print("\nBuilding initial SCRIPT-BPE vocabulary from pre-tokens …")
    all_pretoken_lists = list(lang_pretokens.values())
    vocab = build_initial_vocab_from_pretokens(all_pretoken_lists)   
    # print("\nBuilding initial SCRIPT-BPE vocabulary from pre-tokens …")
    # all_pretoken_lists = list(training_pretokens.values())  # <--- FIXED
    # vocab = build_initial_vocab_from_pretokens(all_pretoken_lists)
    initial_vocab_size = len(vocab)
    print(f"  Initial vocab size: {initial_vocab_size} unique codepoints")

    num_merges = target_vocab_size - initial_vocab_size
    if num_merges <= 0:
        raise ValueError(
            f"Initial vocab ({initial_vocab_size}) ≥ target ({target_vocab_size}). "
            "Reduce Devanagari/Telugu block supplement or increase target vocab size."
        )
    print(f"  Will perform {num_merges} merges to reach {target_vocab_size} tokens.\n")

    # --- Step 6: Normal BPE merge loop with dictionary optimization ---
    # Precompute total word counts for progress logging (NOT unique)
    lang_word_counts = {
        "en": len(corpora["en"].lower().split()),
        "hi": len(corpora["hi"].split()),
        "te": len(corpora["te"].split()),
        "sa": len(corpora["sa"].split()),
    }

    merges = []
    
    # Import dictionary-optimized functions
    from bpe import get_stats_dict, merge_dict

    for step in range(num_merges):
        # Get statistics from training dictionary
        global_stats = get_stats_dict(training_word_freqs)
        if not global_stats:
            print(f"  No more pairs to merge at step {step}. Stopping early.")
            break

        # Normal BPE: pick the globally most frequent pair
        pair = normal_bpe_step(global_stats)
        merged_token = pair[0] + pair[1]
        merges.append(pair)

        # Apply merge to training dictionary
        training_word_freqs = merge_dict(training_word_freqs, pair)
        
        # Apply merge to language dictionaries for fertility tracking
        for lang in LANGUAGES:
            eval_word_freqs[lang] = merge_dict(eval_word_freqs[lang], pair)

        # Add merged token to vocab
        vocab[merged_token] = initial_vocab_size + step

        # Progress logging — show fertility for all languages
        if (step + 1) % PROGRESS_EVERY == 0 or step == num_merges - 1:
            # Calculate fertility from dictionaries
            fertilities = {}
            for lang in LANGUAGES:
                total_tokens = sum(len(word) * freq for word, freq in eval_word_freqs[lang].items())
                fertilities[lang] = total_tokens / lang_word_counts[lang]
                
            fert_str = " | ".join(
                f"{lang}={fertilities[lang]:.3f}" for lang in LANGUAGES
            )
            print(
                f"  Merge {step + 1:>5}/{num_merges} | {fert_str} | "
                f"pair='{pair[0]}'+'{pair[1]}'"
            )

    # --- Step 7: Validate and serialise ---
    assert len(vocab) == target_vocab_size, f"Vocabulary size must be exactly {target_vocab_size}, got {len(vocab)}"
    
    save(vocab, merges, OUTPUT_DIR)
    print(f"\nTraining complete. Vocab: {len(vocab)} tokens. Merges: {len(merges)}.")

    # Print estimated final fertility
    fertilities = {}
    for lang in LANGUAGES:
        total_tokens = sum(len(word) * freq for word, freq in eval_word_freqs[lang].items())
        fertilities[lang] = total_tokens / lang_word_counts[lang]
        
    x_vals = list(fertilities.values())
    print("\nEstimated fertility on training text (final):")
    for lang, x in fertilities.items():
        print(f"  {lang}: {x:.4f}")
    diff = max(x_vals) - min(x_vals)
    if diff > 0:
        print(f"  Score estimate: 1000 / {diff:.4f} = {1000/diff:.1f}")

    # --- Step 7: Write combined tokenizer.json for the widget ---
    combined = {"vocab": vocab, "merges": [list(m) for m in merges]}
    combined_path = os.path.join(OUTPUT_DIR, "tokenizer_combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False)

    widget_data_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "widget", "data"
    )
    os.makedirs(widget_data_dir, exist_ok=True)
    shutil.copy(combined_path, os.path.join(widget_data_dir, "tokenizer.json"))
    print(f"Combined tokenizer copied to widget/data/tokenizer.json")


if __name__ == "__main__":
    train()

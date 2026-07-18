"""
bpe.py — Core BPE engine with regex pre-tokenization support.

Functions:
  get_stats(ids)                        — count adjacent pair frequencies
  get_stats_for_lang(lang_ids, lang)    — pair frequencies for one language
  merge(ids, pair, idx)                 — replace all occurrences of pair with idx
  build_initial_vocab_from_pretokens()  — build vocab from pre-tokenized chunks
  text_to_pretokens()                   — convert text to pre-tokens using regex
  pretokens_to_ids()                    — convert pre-tokens to character IDs  
  normal_bpe_step(global_stats)         — standard BPE merge selection
  parity_aware_step(lang_ids, ...)      — select merge pair for worst-fertility language
"""

import unicodedata
from typing import Union, List, Dict
from regex_pretokenizer import pretokenize_multilingual


# ---------------------------------------------------------------------------
# T008 — pair statistics and merge (ported from SanskritBPETokenizer.ipynb)
# ---------------------------------------------------------------------------

def get_stats(ids: list) -> dict:
    """Count frequency of every adjacent pair in ids."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def get_stats_for_lang(lang_ids: dict, lang: str) -> dict:
    """Count adjacent-pair frequencies in one language's token sequence only."""
    return get_stats(lang_ids[lang])


def merge(ids: list, pair: tuple, idx: int) -> list:
    """
    In ids, replace every non-overlapping occurrence of pair with idx.
    Left-to-right single-pass (standard BPE behaviour).
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def merge_str(tokens: list, pair: tuple) -> list:
    """
    String-level merge: replace every non-overlapping occurrence of
    (left_str, right_str) with left_str + right_str.
    Used during encoding from the exported tokenizer files.
    """
    left, right = pair
    merged = left + right
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
            new_tokens.append(merged)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


def get_stats_dict(word_freqs: Dict[tuple, int]) -> dict:
    """
    Count adjacent pair frequencies across a word-frequency dictionary.
    This is O(V) instead of O(N), where V is unique words and N is total corpus size.
    
    Args:
        word_freqs: Dictionary mapping word tuples (e.g., ('H', 'e', 'l', 'l', 'o')) to frequencies
        
    Returns:
        Dictionary of pair frequencies
    """
    counts = {}
    for word, freq in word_freqs.items():
        if len(word) >= 2:
            for pair in zip(word, word[1:]):
                counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge_dict(word_freqs: Dict[tuple, int], pair: tuple) -> Dict[tuple, int]:
    """
    Apply merge operation to a word-frequency dictionary.
    
    Args:
        word_freqs: Dictionary mapping word tuples to frequencies
        pair: Tuple of (left, right) tokens to merge
        
    Returns:
        New dictionary with merged word tuples
    """
    new_word_freqs = {}
    left, right = pair
    merged = left + right
    
    for word, freq in word_freqs.items():
        if len(word) < 2:
            new_word_freqs[word] = freq
            continue
            
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
                
        new_word_freqs[tuple(new_word)] = new_word_freqs.get(tuple(new_word), 0) + freq
        
    return new_word_freqs


def get_stats_pretokens(pretoken_lists: List[List[str]]) -> dict:
    """
    Count adjacent pair frequencies across lists of pre-tokens.
    Only counts pairs within the same pre-token, never across pre-token boundaries.
    
    Args:
        pretoken_lists: List of lists, where each inner list is a pre-token as characters
        
    Returns:
        Dictionary of pair frequencies
    """
    counts = {}
    for pretoken_chars in pretoken_lists:
        if len(pretoken_chars) >= 2:
            for pair in zip(pretoken_chars, pretoken_chars[1:]):
                counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pretokens(pretoken_lists: List[List[str]], pair: tuple) -> List[List[str]]:
    """
    Apply merge operation to lists of pre-tokens, respecting pre-token boundaries.
    Only merges within individual pre-tokens, never across boundaries.
    
    Args:
        pretoken_lists: List of lists, where each inner list is a pre-token as characters
        pair: Tuple of (left, right) characters to merge
        
    Returns:
        Updated list of pre-token lists with merges applied
    """
    result = []
    for pretoken_chars in pretoken_lists:
        # Apply merge_str to this individual pre-token
        merged_pretoken = merge_str(pretoken_chars, pair)
        result.append(merged_pretoken)
    return result


# ---------------------------------------------------------------------------
# Pre-tokenization functions for regex-based chunking
# ---------------------------------------------------------------------------

def text_to_pretokens(text: str, language: str) -> List[str]:
    """
    Convert raw text to pre-tokenized chunks using language-specific regex patterns.
    
    Args:
        text: Raw input text
        language: Language code ('en', 'hi', 'te', 'sa')
        
    Returns:
        List of pre-tokenized chunks ready for character-level BPE
    """
    return pretokenize_multilingual(text, language)


def pretokens_to_ids(pretokens: List[str], vocab: dict) -> List[int]:
    """
    Convert pre-tokenized chunks to character-level IDs for BPE processing.
    Each pre-token is split into its constituent characters, then mapped to vocab IDs.
    
    Args:
        pretokens: List of pre-tokenized text chunks
        vocab: Character-to-ID mapping
        
    Returns:
        List of character IDs suitable for BPE merging
    """
    ids = []
    for pretoken in pretokens:
        # Convert each character in the pre-token to its vocab ID
        for char in pretoken:
            if char in vocab:
                ids.append(vocab[char])
            else:
                # Handle unknown characters - this shouldn't happen with proper vocab building
                print(f"Warning: Unknown character '{char}' (U+{ord(char):04X}) in pre-token '{pretoken}'")
                # Could add to vocab dynamically or use special unknown token
    return ids


def ids_to_pretokens(ids: List[int], vocab: dict) -> List[str]:
    """
    Convert character IDs back to pre-tokenized chunks for validation/debugging.
    
    Args:
        ids: List of character IDs
        vocab: Character-to-ID mapping
        
    Returns:
        List reconstructed from the character IDs
    """
    # Create reverse vocab mapping
    id_to_char = {idx: char for char, idx in vocab.items()}
    
    # Convert IDs back to characters
    chars = []
    for id_val in ids:
        if id_val in id_to_char:
            chars.append(id_to_char[id_val])
        else:
            chars.append('<?>')  # Unknown ID marker
    
    return [''.join(chars)]  # Return as single string for now


# ---------------------------------------------------------------------------
# T009 — SCRIPT-BPE initial vocabulary (Unicode codepoints, not bytes)
# ---------------------------------------------------------------------------

# Devanagari block: U+0900–U+097F
_DEVANAGARI = [chr(cp) for cp in range(0x0900, 0x0980)]
# Telugu block: U+0C00–U+0C7F
_TELUGU = [chr(cp) for cp in range(0x0C00, 0x0C80)]


def build_initial_vocab_from_pretokens(all_pretokens: List[List[str]]) -> dict:
    """
    Build the initial character-level vocabulary from pre-tokenized training corpus.

    Strategy (SCRIPT-BPE with regex pre-tokenization):
      1. Collect all unique Unicode codepoints that appear in all pre-tokenized chunks.
      2. Supplement with the complete Devanagari and Telugu Unicode blocks
         so that rare characters encountered at inference time are still covered.
      3. Sort for determinism, then assign sequential integer IDs starting at 0.

    Args:
        all_pretokens: List of pre-tokenized chunk lists for all languages
        
    Returns:
        vocab: dict mapping character → integer ID
    """
    chars = set()
    
    # Collect all characters from all pre-tokenized chunks
    for pretoken_list in all_pretokens:
        for pretoken in pretoken_list:
            chars.update(pretoken)
    
    # Supplement with full Devanagari and Telugu blocks
    chars.update(_DEVANAGARI)
    chars.update(_TELUGU)
    
    # Supplement with all printable ASCII characters to ensure roundtrip fidelity
    # for URLs and Markdown formatting that might not appear in the training corpus
    import string
    chars.update(string.printable)

    # Sort: printable ASCII first, then by Unicode code point
    sorted_chars = sorted(chars, key=lambda c: (ord(c) > 127, ord(c)))
    vocab = {ch: idx for idx, ch in enumerate(sorted_chars)}

    if len(vocab) >= 10_000:
        raise ValueError(
            f"Initial vocabulary ({len(vocab)} tokens) already ≥ 10,000. "
            "Cannot add any merge tokens. Check corpus or reduce Devanagari/Telugu block supplement."
        )
    return vocab


def build_initial_vocab(all_text: str) -> dict:
    """
    Legacy function for backward compatibility.
    Build the initial character-level vocabulary from raw training text.
    
    For regex pre-tokenization, use build_initial_vocab_from_pretokens() instead.
    """
    chars = set(all_text)
    # Supplement with full Devanagari and Telugu blocks
    chars.update(_DEVANAGARI)
    chars.update(_TELUGU)

    # Sort: printable ASCII first, then by Unicode code point
    sorted_chars = sorted(chars, key=lambda c: (ord(c) > 127, ord(c)))
    vocab = {ch: idx for idx, ch in enumerate(sorted_chars)}

    if len(vocab) >= 10_000:
        raise ValueError(
            f"Initial vocabulary ({len(vocab)} tokens) already ≥ 10,000. "
            "Cannot add any merge tokens. Check corpus or reduce Devanagari/Telugu block supplement."
        )
    return vocab


# ---------------------------------------------------------------------------
# Parity-Aware merge step (kept as optional; not used for standard training)
# ---------------------------------------------------------------------------

def compute_fertility(lang_token_lists: dict, lang_word_counts: dict) -> dict:
    """
    Compute current fertility ratio for each language (training-time estimate).
    fertility_i = total_tokens / total_word_count_i
    
    Handles both flat token lists and pre-token lists (list of lists).
    """
    result = {}
    for lang in lang_token_lists:
        token_data = lang_token_lists[lang]
        
        # Check if this is a list of pre-token lists or a flat list
        if token_data and isinstance(token_data[0], list):
            # Pre-token lists: sum up all tokens across all pre-tokens
            total_tokens = sum(len(pretoken) for pretoken in token_data)
        else:
            # Flat list: just count the tokens
            total_tokens = len(token_data)
        
        result[lang] = total_tokens / max(lang_word_counts[lang], 1)
    
    return result


def normal_bpe_step(global_stats: dict) -> tuple:
    """
    Standard BPE merge selection: return the globally most-frequent adjacent pair.
    This is the classic Sennrich et al. (2016) algorithm.

    Args:
        global_stats: pre-computed global pair frequency dict

    Returns:
        (left_token, right_token) — the most frequent pair to merge next
    """
    if not global_stats:
        raise StopIteration("No pairs left to merge.")
    return max(global_stats, key=global_stats.get)


def parity_aware_step(
    lang_ids: dict,
    lang_word_counts: dict,
    global_stats: dict,
) -> tuple:
    """
    Select the BPE merge pair using the Parity-Aware strategy (Foroutan et al., 2025).
    OPTIONAL — not used in the default training pipeline.

    Algorithm:
      1. Identify the language with the highest current fertility ratio.
      2. Compute pair statistics for that language only.
      3. Return the most-frequent pair within that language's token sequence.

    Falls back to the globally most-frequent pair if the worst language
    has no pairs (e.g., single-token corpus).
    """
    fertilities = compute_fertility(lang_ids, lang_word_counts)
    worst_lang = max(fertilities, key=fertilities.get)

    lang_stats = get_stats_for_lang(lang_ids, worst_lang)
    if lang_stats:
        return max(lang_stats, key=lang_stats.get)

    # Fallback: globally most-frequent pair
    if global_stats:
        return max(global_stats, key=global_stats.get)

    raise StopIteration("No pairs left to merge.")

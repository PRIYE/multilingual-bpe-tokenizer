"""
Language-specific regex pre-tokenization for multilingual BPE.

This module provides regex patterns and pre-tokenization functions for English, Hindi, 
Telugu, and Sanskrit. Based on research from MUTANT-Indic, BrahmicTokenizer, and 
IndicSuperTokenizer papers showing that LLaMA-4 style regex outperforms GPT-2 
by 38-40% on Indic scripts.

Key principles:
1. Use LLaMA-4/Cl100k style patterns for better Indic script handling
2. Apply Unicode normalization (NFKC) to reduce sparsity
3. Handle language-specific punctuation and structures
4. Preserve morphological boundaries within pre-tokens
"""

import re
import regex
import unicodedata
import os
from typing import List, Tuple

# Try to import Indic normalization, fallback if not available
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    INDIC_NORMALIZE_AVAILABLE = True
except ImportError:
    IndicNormalizerFactory = None
    INDIC_NORMALIZE_AVAILABLE = False
    print("Warning: indic_nlp library not available. Indic normalization disabled.")

# Try to import Indic morphology (Morfessor) for Sandhi splitting
try:
    from indicnlp import common
    from indicnlp.morph import unsupervised_morph
    
    # Set resources path to the downloaded folder
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    resources_path = os.path.join(workspace_root, "indic_nlp_resources-master")
    
    if os.path.exists(resources_path):
        common.INDIC_RESOURCES_PATH = resources_path
        
        # Initialize analyzers
        MORPH_ANALYZERS = {
            'hi': unsupervised_morph.UnsupervisedMorphAnalyzer('hi'),
            'te': unsupervised_morph.UnsupervisedMorphAnalyzer('te'),
            'sa': unsupervised_morph.UnsupervisedMorphAnalyzer('sa')
        }
    else:
        MORPH_ANALYZERS = {}
        print("Warning: indic_nlp_resources-master not found. Sandhi splitting disabled.")
except Exception as e:
    MORPH_ANALYZERS = {}
    print(f"Warning: Failed to load morph analyzers ({e}). Sandhi splitting disabled.")


# Base LLaMA-4/Cl100k regex pattern (outperforms GPT-2 on Indic scripts)
LLAMA4_BASE_PATTERN = r"""(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# GPT-2 pattern for comparison
GPT2_BASE_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def normalize_unicode(text: str) -> str:
    """
    Apply NFKC normalization to reduce orthographic sparsity.
    Essential for Indic scripts where characters can be represented multiple ways.
    """
    return unicodedata.normalize('NFKC', text)


def apply_morfessor_and_akshara(text: str, lang: str, akshara_pattern: str, dev_range: str) -> List[str]:
    """
    Two-stage pre-tokenization for Indic languages:
    1. Split text into words (preserving spaces and punctuation)
    2. Pass valid Indic words to Morfessor for Sandhi/Morphological splitting
    3. Return the resulting morphemes (without Akshara splitting)
    """
    analyzer = MORPH_ANALYZERS.get(lang)
    
    # If no analyzer, just fall back to word-level regex directly
    if not analyzer:
        word_pattern = f'[।॥॰]+| ?{dev_range}+| ?[0-9०-९]{{1,3}}| ?[^\\s\u0900-\u097F\u0C00-\u0C7F0-9०-९।॥॰]+|\\s+(?!\\S)|\\s+'
        return [m for m in regex.findall(word_pattern, text) if m]
        
    # Stage 1: Word-level splitting
    # Grabs words with leading spaces, numbers, punctuation, and other symbols
    word_pattern = f'[।॥॰]+| ?{dev_range}+| ?[0-9०-९]{{1,3}}| ?[^\\s\u0900-\u097F\u0C00-\u0C7F0-9०-९।॥॰]+|\\s+(?!\\S)|\\s+'
    words = regex.findall(word_pattern, text)
    
    final_tokens = []
    for word in words:
        # Check if it's a pure Indic word (ignoring leading space)
        if regex.match(f'^ ?{dev_range}+$', word):
            leading_space = ""
            clean_word = word
            if word.startswith(" "):
                leading_space = " "
                clean_word = word[1:]
                
            # Morph analyze
            try:
                morphemes = analyzer.morph_analyze(clean_word)
            except:
                morphemes = [clean_word]
            
            # Add morphemes directly (no Akshara splitting)
            for i, morph in enumerate(morphemes):
                if i == 0:
                    morph = leading_space + morph
                final_tokens.append(morph)
        else:
            # Non-Indic word, keep as is
            final_tokens.append(word)
            
    return final_tokens


def english_pretokenize(text: str) -> List[str]:
    """
    English pre-tokenization using standard GPT-2 style pattern.
    This groups spaces with words to allow BPE to merge them efficiently,
    which is critical for achieving a low fertility ratio.
    
    Args:
        text: Raw English text
        
    Returns:
        List of pre-tokenized chunks
    """
    # Apply Unicode normalization
    text = normalize_unicode(text)
    
    # Use standard GPT-2 pattern that prepends spaces to words
    pattern = GPT2_BASE_PATTERN
    
    # Use regex module for \p{L} support
    matches = regex.findall(pattern, text)
    
    # Filter empty matches but preserve whitespace for reversibility
    return [match for match in matches if match]


def hindi_pretokenize(text: str) -> List[str]:
    """
    Hindi pre-tokenization using Sandhi splitting.
    Splits words into constituent morphemes to prevent BPE from 
    wasting merges on massive compound words.
    
    Args:
        text: Raw Hindi text in Devanagari script
        
    Returns:
        List of pre-tokenized chunks
    """
    # Apply Unicode normalization
    text = normalize_unicode(text)
    
    # Apply Indic normalization if available
    if INDIC_NORMALIZE_AVAILABLE:
        try:
            normalizer = IndicNormalizerFactory().get_normalizer("hi")
            text = normalizer.normalize(text)
        except:
            pass  # Continue without Indic normalization if it fails
    
    dev_range = r"[\u0900-\u0963\u0966-\u097F]"
    
    return apply_morfessor_and_akshara(text, 'hi', "", dev_range)


def telugu_pretokenize(text: str) -> List[str]:
    """
    Telugu pre-tokenization using Sandhi splitting.
    Splits words into constituent morphemes to prevent BPE from 
    wasting merges on massive compound words.
    
    Args:
        text: Raw Telugu text
        
    Returns:
        List of pre-tokenized chunks
    """
    # Apply Unicode normalization
    text = normalize_unicode(text)
    
    # Apply Indic normalization if available
    if INDIC_NORMALIZE_AVAILABLE:
        try:
            normalizer = IndicNormalizerFactory().get_normalizer("te")
            text = normalizer.normalize(text)
        except:
            pass
    
    dev_range = r"[\u0C00-\u0C7F]"
    
    return apply_morfessor_and_akshara(text, 'te', "", dev_range)


def sanskrit_pretokenize(text: str) -> List[str]:
    """
    Sanskrit pre-tokenization using Sandhi splitting.
    Splits words into constituent morphemes to prevent BPE from 
    wasting merges on massive compound words.
    
    Args:
        text: Raw Sanskrit text in Devanagari script
        
    Returns:
        List of pre-tokenized chunks
    """
    # Apply Unicode normalization
    text = normalize_unicode(text)
    
    # Apply Indic normalization if available
    if INDIC_NORMALIZE_AVAILABLE:
        try:
            normalizer = IndicNormalizerFactory().get_normalizer("sa")
            text = normalizer.normalize(text)
        except:
            pass
    
    dev_range = r"[\u0900-\u0963\u0966-\u097F]"
    
    return apply_morfessor_and_akshara(text, 'sa', "", dev_range)


def get_language_pretokenizer(language: str):
    """
    Get the appropriate pre-tokenization function for a language.
    
    Args:
        language: Language code ('en', 'hi', 'te', 'sa')
        
    Returns:
        Pre-tokenization function for the specified language
    """
    pretokenizers = {
        'en': english_pretokenize,
        'hi': hindi_pretokenize, 
        'te': telugu_pretokenize,
        'sa': sanskrit_pretokenize
    }
    
    return pretokenizers.get(language, english_pretokenize)


def pretokenize_multilingual(text: str, language: str) -> List[str]:
    """
    Apply language-specific pre-tokenization to text.
    
    Args:
        text: Input text
        language: Language code ('en', 'hi', 'te', 'sa')
        
    Returns:
        List of pre-tokenized chunks ready for BPE
    """
    pretokenizer = get_language_pretokenizer(language)
    return pretokenizer(text)


# Test function
if __name__ == "__main__":
    # Test with sample texts
    test_cases = [
        ("Hello world! How are you?", "en"),
        ("नमस्ते दुनिया। आप कैसे हैं?", "hi"),
        ("హలో ప్రపంచం! మీరు ఎలా ఉన్నారు?", "te"),
        ("ॐ तत्सत्। सर्वं खल्विदं ब्रह्म।", "sa")
    ]
    
    for text, lang in test_cases:
        tokens = pretokenize_multilingual(text, lang)
        print(f"{lang}: {text}")
        print(f"Tokens: {tokens}")
        print()
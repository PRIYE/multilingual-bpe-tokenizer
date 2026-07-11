import unittest
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from regex_pretokenizer import (
    english_pretokenize,
    hindi_pretokenize,
    telugu_pretokenize,
    sanskrit_pretokenize,
    pretokenize_multilingual
)


class TestRegexPretokenizer(unittest.TestCase):
    
    def test_whitespace_preservation(self):
        """Test that whitespace is preserved in tokens to ensure reversibility."""
        text = "Hello world! How are you?"
        tokens = english_pretokenize(text)
        # Reconstructing should exactly match the original text
        self.assertEqual("".join(tokens), text)
        
        # Check specific tokens (GPT-2 style prepends space)
        self.assertIn(" world", tokens)
        self.assertIn("!", tokens)

    def test_english_contractions(self):
        """Test that English contractions are handled properly."""
        text = "I'm sure they'll do it, shouldn't they?"
        tokens = english_pretokenize(text)
        self.assertEqual("".join(tokens), text)
        
        # With GPT-2 split, contractions are split
        self.assertIn("'m", tokens)
        self.assertIn("'ll", tokens)

    def test_hindi_punctuation(self):
        """Test Devanagari specific punctuation like danda."""
        text = "नमस्ते दुनिया। आप कैसे हैं॥"
        tokens = hindi_pretokenize(text)
        self.assertEqual("".join(tokens), text)
        
        # Danda should be kept with the word or as a separate token depending on regex
        # Our regex `[।॥]+` matches danda.
        danda_tokens = [t for t in tokens if '।' in t or '॥' in t]
        self.assertTrue(len(danda_tokens) > 0)

    def test_telugu_script(self):
        """Test Telugu script tokenization."""
        text = "హలో ప్రపంచం! మీరు ఎలా ఉన్నారు?"
        tokens = telugu_pretokenize(text)
        self.assertEqual("".join(tokens), text)
        self.assertIn(" ప్రపంచం", tokens)

    def test_sanskrit_sandhi_conjuncts(self):
        """Test Sanskrit specific regex logic."""
        text = "ॐ तत्सत्। सर्वं खल्विदं ब्रह्म।"
        tokens = sanskrit_pretokenize(text)
        self.assertEqual("".join(tokens), text)

    def test_multilingual_router(self):
        """Test that the router calls the correct function."""
        en_tokens = pretokenize_multilingual("Hello", "en")
        hi_tokens = pretokenize_multilingual("नमस्ते", "hi")
        
        self.assertEqual("".join(en_tokens), "Hello")
        self.assertEqual("".join(hi_tokens), "नमस्ते")


if __name__ == "__main__":
    unittest.main()
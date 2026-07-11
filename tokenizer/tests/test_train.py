import unittest
import sys
import os
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from train import apply_language_multipliers, apply_language_multipliers_text, maybe_augment_sanskrit, maybe_augment_telugu


class TestTrain(unittest.TestCase):
    
    def test_apply_language_multipliers_text(self):
        """Test manual multipliers with text."""
        corpora = {
            "en": "Hello world ",
            "hi": "नमस्ते "
        }
        multipliers = {"en": 1, "hi": 3}
        
        sampled = apply_language_multipliers_text(corpora, multipliers)
        
        # Should contain text from both languages
        self.assertIn("Hello", sampled)
        self.assertIn("नमस्ते", sampled)
        
        # Hindi should appear 3 times, English 1 time
        self.assertEqual(sampled.count("Hello world "), 1)
        self.assertEqual(sampled.count("नमस्ते "), 3)

    def test_apply_language_multipliers(self):
        """Test manual multipliers on pre-tokenized lists."""
        pretokens_dict = {
            "en": ["Hello", " world", "!"],
            "hi": ["नमस्ते", " ", "दुनिया"]
        }
        multipliers = {"en": 1, "hi": 2}
        
        sampled = apply_language_multipliers(pretokens_dict, multipliers)
        
        # Should contain pre-tokens from both languages
        sampled_text = "".join(sampled)
        self.assertIn("Hello", sampled_text)
        self.assertIn("नमस्ते", sampled_text)
        
        # English should have 3 tokens * 1 = 3
        # Hindi should have 3 tokens * 2 = 6
        # Total = 9 tokens
        self.assertEqual(len(sampled), 9)
        
        # Hindi tokens should appear twice as often
        self.assertEqual(sampled.count("नमस्ते"), 2)
        self.assertEqual(sampled.count("Hello"), 1)

    def test_apply_language_multipliers_empty(self):
        """Test manual multipliers with empty pre-tokens."""
        pretokens_dict = {
            "en": [],
            "hi": ["नमस्ते"]
        }
        multipliers = {"en": 1, "hi": 1}
        
        sampled = apply_language_multipliers(pretokens_dict, multipliers)
        # Should handle empty lists gracefully
        self.assertEqual(sampled, ["नमस्ते"])

    def test_maybe_augment_sanskrit_short(self):
        """Test Sanskrit augmentation when corpus is too short."""
        short_text = "ॐ"  # Very short Sanskrit text
        
        # This should attempt augmentation but may fail due to network/dataset issues
        # We just test that it returns a string and doesn't crash
        result = maybe_augment_sanskrit(short_text, min_chars=100)
        self.assertIsInstance(result, str)
        self.assertIn("ॐ", result)  # Original text should be preserved

    def test_maybe_augment_sanskrit_long_enough(self):
        """Test Sanskrit augmentation when corpus is already long enough."""
        long_text = "ॐ तत्सत् " * 1000  # Long enough text
        
        result = maybe_augment_sanskrit(long_text, min_chars=100)
        # Should return original text unchanged
        self.assertEqual(result, long_text)

    def test_maybe_augment_telugu_short(self):
        """Test Telugu augmentation when corpus is too short."""
        short_text = "హలో"  # Very short Telugu text
        
        # This should attempt augmentation but may fail due to network/dataset issues
        result = maybe_augment_telugu(short_text, min_chars=1000)
        self.assertIsInstance(result, str)
        self.assertIn("హలో", result)  # Original text should be preserved

    def test_maybe_augment_telugu_long_enough(self):
        """Test Telugu augmentation when corpus is already long enough."""
        long_text = "హలో ప్రపంచం " * 2000  # Long enough text
        
        result = maybe_augment_telugu(long_text, min_chars=1000)
        # Should return original text unchanged
        self.assertEqual(result, long_text)


if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bpe import (
    get_stats_pretokens,
    merge_pretokens,
    compute_fertility,
    build_initial_vocab_from_pretokens
)


class TestBPE(unittest.TestCase):
    
    def setUp(self):
        # Sample pre-tokens represented as lists of characters
        self.sample_pretokens = [
            ['H', 'e', 'l', 'l', 'o'],
            [' ', 'w', 'o', 'r', 'l', 'd']
        ]

    def test_get_stats_pretokens(self):
        """Test that stats are collected within boundaries, not across."""
        stats = get_stats_pretokens(self.sample_pretokens)
        
        # ('e', 'l') should be 1
        self.assertEqual(stats.get(('e', 'l')), 1)
        # ('l', 'l') should be 1
        self.assertEqual(stats.get(('l', 'l')), 1)
        
        # Cross-boundary pair ('o', ' ') should NOT exist
        self.assertNotIn(('o', ' '), stats)

    def test_merge_pretokens(self):
        """Test that merges respect boundaries."""
        # Merge 'l' and 'l'
        merged = merge_pretokens(self.sample_pretokens, ('l', 'l'))
        
        # First word should now be ['H', 'e', 'll', 'o']
        self.assertEqual(merged[0], ['H', 'e', 'll', 'o'])
        # Second word unchanged
        self.assertEqual(merged[1], [' ', 'w', 'o', 'r', 'l', 'd'])

    def test_compute_fertility(self):
        """Test fertility calculation for pre-token lists."""
        lang_tokens = {
            "en": [['H', 'e', 'll', 'o'], [' ', 'w', 'o', 'r', 'l', 'd']]
        }
        lang_word_counts = {"en": 2}
        
        fertility = compute_fertility(lang_tokens, lang_word_counts)
        
        # 4 tokens in first word, 6 in second = 10 tokens total. 10 / 2 = 5.0
        self.assertEqual(fertility["en"], 5.0)

    def test_build_initial_vocab(self):
        """Test vocabulary building from pre-tokens."""
        vocab = build_initial_vocab_from_pretokens([self.sample_pretokens])
        
        # Should contain all unique characters
        self.assertIn('H', vocab)
        self.assertIn('e', vocab)
        self.assertIn(' ', vocab)
        
        # Should contain Devanagari and Telugu supplements
        self.assertIn(chr(0x0905), vocab) # Devanagari 'अ'
        self.assertIn(chr(0x0C05), vocab) # Telugu 'అ'


if __name__ == "__main__":
    unittest.main()
import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bpe import (
    get_stats,
    merge_str,
    get_stats_dict,
    merge_dict,
    normal_bpe_step,
    parity_aware_step,
    text_to_pretokens,
    pretokens_to_ids,
    ids_to_pretokens,
    build_initial_vocab
)


class TestBPEExtended(unittest.TestCase):
    
    def test_get_stats_basic(self):
        """Test basic get_stats function."""
        tokens = ['H', 'e', 'l', 'l', 'o']
        stats = get_stats(tokens)
        
        self.assertEqual(stats[('H', 'e')], 1)
        self.assertEqual(stats[('l', 'l')], 1)
        self.assertEqual(stats[('e', 'l')], 1)

    def test_get_stats_empty(self):
        """Test get_stats with empty or single token."""
        self.assertEqual(get_stats([]), {})
        self.assertEqual(get_stats(['H']), {})

    def test_merge_str_basic(self):
        """Test basic merge_str function."""
        tokens = ['H', 'e', 'l', 'l', 'o']
        merged = merge_str(tokens, ('l', 'l'))
        
        self.assertEqual(merged, ['H', 'e', 'll', 'o'])

    def test_merge_str_no_matches(self):
        """Test merge_str when pair doesn't exist."""
        tokens = ['H', 'e', 'l', 'o']
        merged = merge_str(tokens, ('x', 'y'))
        
        # Should return unchanged
        self.assertEqual(merged, tokens)

    def test_merge_str_multiple_matches(self):
        """Test merge_str with multiple occurrences."""
        tokens = ['a', 'b', 'c', 'a', 'b', 'd']
        merged = merge_str(tokens, ('a', 'b'))
        
        self.assertEqual(merged, ['ab', 'c', 'ab', 'd'])

    def test_get_stats_dict(self):
        """Test get_stats_dict function."""
        word_freqs = {
            ('H', 'e', 'l', 'l', 'o'): 2,
            ('w', 'o', 'r', 'l', 'd'): 1
        }
        stats = get_stats_dict(word_freqs)
        
        # ('l', 'l') appears once in Hello, which has freq 2
        self.assertEqual(stats[('l', 'l')], 2)
        # ('o', 'r') appears once in world, which has freq 1
        self.assertEqual(stats[('o', 'r')], 1)
        
    def test_merge_dict(self):
        """Test merge_dict function."""
        word_freqs = {
            ('H', 'e', 'l', 'l', 'o'): 2,
            ('w', 'o', 'r', 'l', 'd'): 1
        }
        merged = merge_dict(word_freqs, ('l', 'l'))
        
        # Check that Hello was merged
        self.assertIn(('H', 'e', 'll', 'o'), merged)
        self.assertEqual(merged[('H', 'e', 'll', 'o')], 2)
        
        # Check that world was unchanged
        self.assertIn(('w', 'o', 'r', 'l', 'd'), merged)
        self.assertEqual(merged[('w', 'o', 'r', 'l', 'd')], 1)
        
    def test_normal_bpe_step(self):
        """Test normal BPE step selection."""
        stats = {('a', 'b'): 3, ('b', 'c'): 1, ('c', 'd'): 5}
        
        pair = normal_bpe_step(stats)
        # Should return the most frequent pair
        self.assertEqual(pair, ('c', 'd'))

    def test_normal_bpe_step_empty(self):
        """Test normal BPE step with empty stats."""
        with self.assertRaises(StopIteration):
            normal_bpe_step({})

    def test_parity_aware_step_basic(self):
        """Test parity-aware BPE step."""
        lang_ids = {
            "en": ['H', 'e', 'l', 'l', 'o'],
            "hi": ['न', 'म', 'स', 'त', 'े']
        }
        lang_word_counts = {"en": 1, "hi": 1}
        global_stats = {('H', 'e'): 1, ('e', 'l'): 1, ('l', 'l'): 1, ('l', 'o'): 1, 
                       ('न', 'म'): 1, ('म', 'स'): 1, ('स', 'त'): 1, ('त', 'े'): 1}
        
        # Should pick pair from language with highest fertility (both have same fertility, so either is valid)
        pair = parity_aware_step(lang_ids, lang_word_counts, global_stats)
        # Should be a valid pair from the global stats
        self.assertIn(pair, global_stats.keys())

    def test_text_to_pretokens(self):
        """Test text to pre-tokens conversion."""
        text = "Hello world"
        pretokens = text_to_pretokens(text, "en")
        
        # Should be a list of strings
        self.assertIsInstance(pretokens, list)
        self.assertTrue(all(isinstance(token, str) for token in pretokens))
        # Should reconstruct original text
        self.assertEqual("".join(pretokens), text)

    def test_pretokens_to_ids(self):
        """Test pre-tokens to IDs conversion."""
        pretokens = ["Hello", " world"]
        vocab = {c: i for i, c in enumerate("Helo wrld")}
        
        ids = pretokens_to_ids(pretokens, vocab)
        
        # Should be a list of integers
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(id_val, int) for id_val in ids))

    def test_pretokens_to_ids_unknown_char(self):
        """Test pre-tokens to IDs with unknown characters."""
        pretokens = ["Hello", " xyz"]  # 'x', 'y', 'z' not in vocab
        vocab = {'H': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4}
        
        # Should skip unknown characters gracefully
        ids = pretokens_to_ids(pretokens, vocab)
        expected_ids = [0, 1, 2, 2, 3, 4]  # H, e, l, l, o, space (xyz skipped)
        self.assertEqual(ids, expected_ids)

    def test_ids_to_pretokens(self):
        """Test IDs to pre-tokens conversion."""
        ids = [0, 1, 2, 2, 3]
        vocab = {'H': 0, 'e': 1, 'l': 2, 'o': 3}
        
        result = ids_to_pretokens(ids, vocab)
        
        # Should be a list containing the reconstructed text
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["Hello"])

    def test_build_initial_vocab_legacy(self):
        """Test legacy initial vocabulary building."""
        text = "Hello नमस्ते"
        vocab = build_initial_vocab(text)
        
        # Should contain all unique characters
        self.assertIn('H', vocab)
        self.assertIn('न', vocab)
        # Should contain Devanagari and Telugu blocks
        self.assertIn(chr(0x0905), vocab)
        self.assertIn(chr(0x0C05), vocab)


if __name__ == "__main__":
    unittest.main()
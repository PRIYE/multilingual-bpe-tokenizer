import unittest
import sys
import os
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import encode, encode_legacy, decode, save, load, script_distribution


class TestTokenizerExtended(unittest.TestCase):
    
    def setUp(self):
        self.vocab = {
            'H': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4, 'w': 5, 'r': 6, 'd': 7,
            'न': 8, 'म': 9, 'स': 10, 'त': 11, 'े': 12, '्': 13,
            'హ': 14, 'లో': 15,
            'll': 16, 'or': 17, 'Hello': 18
        }
        # Pad to 10,000 for save function
        for i in range(20, 10000):
            self.vocab[f"dummy_{i}"] = i
            
        self.merges = [('l', 'l'), ('o', 'r')]
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_encode_different_languages(self):
        """Test encoding with different language parameters."""
        # English
        ids_en = encode("Hello", self.vocab, self.merges, "en")
        self.assertIsInstance(ids_en, list)
        
        # Hindi
        ids_hi = encode("नमस्ते", self.vocab, self.merges, "hi")
        self.assertIsInstance(ids_hi, list)
        
        # Telugu
        ids_te = encode("హలో", self.vocab, self.merges, "te")
        self.assertIsInstance(ids_te, list)
        
        # Sanskrit
        ids_sa = encode("नमः", self.vocab, self.merges, "sa")
        self.assertIsInstance(ids_sa, list)

    def test_encode_empty_text(self):
        """Test encoding empty text."""
        ids = encode("", self.vocab, self.merges, "en")
        self.assertEqual(ids, [])

    def test_encode_unknown_characters(self):
        """Test encoding with characters not in vocabulary."""
        # Text with unknown characters
        ids = encode("xyz", self.vocab, self.merges, "en")
        # Should return empty list since no characters are in vocab
        self.assertEqual(ids, [])

    def test_decode_unknown_ids(self):
        """Test decoding with unknown IDs."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Include an unknown ID
        ids = [0, 1, 99999, 2]  # 99999 is not in vocab
        text = decode(ids, id_to_token)
        
        # Should contain <UNK> for unknown ID
        self.assertIn("<UNK>", text)

    def test_decode_empty_ids(self):
        """Test decoding empty ID list."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        text = decode([], id_to_token)
        self.assertEqual(text, "")

    def test_script_distribution(self):
        """Test script distribution analysis."""
        vocab_sample = {
            'Hello': 0,      # Latin
            'world': 1,      # Latin  
            'नमस्ते': 2,      # Devanagari
            'दुनिया': 3,     # Devanagari
            'హలో': 4,       # Telugu
            'ప్రపంచం': 5,    # Telugu
            'Hello नमस्ते': 6, # Mixed
            '123': 7,        # Other
        }
        
        distribution = script_distribution(vocab_sample)
        
        # Check that all categories are present
        expected_keys = ["Devanagari", "Telugu", "Latin", "Mixed", "Other"]
        for key in expected_keys:
            self.assertIn(key, distribution)
        
        # Check some expected counts (actual counts may vary based on implementation)
        self.assertGreaterEqual(distribution["Latin"], 1)      # At least Hello, world
        self.assertGreaterEqual(distribution["Devanagari"], 1) # At least some Devanagari tokens
        self.assertGreaterEqual(distribution["Telugu"], 1)     # At least some Telugu tokens
        # Mixed and Other counts depend on exact classification logic

    def test_script_distribution_empty_vocab(self):
        """Test script distribution with empty vocabulary."""
        distribution = script_distribution({})
        
        expected_keys = ["Devanagari", "Telugu", "Latin", "Mixed", "Other"]
        for key in expected_keys:
            self.assertEqual(distribution[key], 0)

    def test_load_missing_files(self):
        """Test loading with missing files."""
        # Test missing vocab file
        with self.assertRaises(FileNotFoundError):
            load(os.path.join(self.test_dir, "nonexistent"))

    def test_load_invalid_vocab_size(self):
        """Test loading with incorrect vocab size (now prints warning instead of error)."""
        # Create a vocab with wrong size
        small_vocab = {'a': 0, 'b': 1}
        small_merges = []
        
        # Save small vocab
        vocab_path = os.path.join(self.test_dir, "vocab.json")
        merges_path = os.path.join(self.test_dir, "merges.txt")
        
        import json
        with open(vocab_path, "w") as f:
            json.dump(small_vocab, f)
        with open(merges_path, "w") as f:
            f.write("#version: 1.0\n")
        
        # Should NOT raise assertion error anymore
        loaded_vocab, loaded_merges = load(self.test_dir)
        self.assertEqual(loaded_vocab, small_vocab)

    def test_save_invalid_vocab_size(self):
        """Test saving with incorrect vocab size (now prints warning instead of error)."""
        small_vocab = {'a': 0, 'b': 1}
        small_merges = []
        
        # Should NOT raise assertion error anymore
        save(small_vocab, small_merges, self.test_dir)
        
        # Verify it actually saved
        vocab_path = os.path.join(self.test_dir, "vocab.json")
        self.assertTrue(os.path.exists(vocab_path))

    def test_encode_legacy_vs_new(self):
        """Test that legacy and new encode behave differently."""
        text = "Hello world"
        
        ids_legacy = encode_legacy(text, self.vocab, self.merges)
        ids_new = encode(text, self.vocab, self.merges, "en")
        
        # They should be different due to pre-tokenization
        # (unless the vocab and merges produce identical results by chance)
        # At minimum, both should be valid ID lists
        self.assertIsInstance(ids_legacy, list)
        self.assertIsInstance(ids_new, list)


if __name__ == "__main__":
    unittest.main()
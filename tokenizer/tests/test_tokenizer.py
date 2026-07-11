import unittest
import sys
import os
import tempfile
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from tokenizer import encode, encode_legacy, decode, save, load


class TestTokenizer(unittest.TestCase):
    
    def setUp(self):
        # Create a dummy vocab and merges
        self.vocab = {
            'H': 0, 'e': 1, 'l': 2, 'o': 3, ' ': 4, 'w': 5, 'r': 6, 'd': 7,
            'll': 8, 'or': 9
        }
        # Pad vocab to 10,000 to pass the save() assertion
        for i in range(10, 10000):
            self.vocab[f"dummy_{i}"] = i
            
        self.merges = [
            ('l', 'l'),
            ('o', 'r')
        ]
        
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_encode_legacy(self):
        """Test legacy character-level encode."""
        text = "Hello"
        # 'H', 'e', 'l', 'l', 'o' -> 'H', 'e', 'll', 'o'
        ids = encode_legacy(text, self.vocab, self.merges)
        expected_ids = [self.vocab['H'], self.vocab['e'], self.vocab['ll'], self.vocab['o']]
        self.assertEqual(ids, expected_ids)

    def test_encode_with_pretokenization(self):
        """Test new encode with regex pre-tokenization."""
        text = "Hello world"
        # Pre-tokens: ['Hello', ' world']
        # 'Hello' -> 'H', 'e', 'll', 'o'
        # ' world' -> ' ', 'w', 'or', 'l', 'd'
        ids = encode(text, self.vocab, self.merges, language="en")
        
        expected_ids = [
            self.vocab['H'], self.vocab['e'], self.vocab['ll'], self.vocab['o'],
            self.vocab[' '], self.vocab['w'], self.vocab['or'], self.vocab['l'], self.vocab['d']
        ]
        self.assertEqual(ids, expected_ids)

    def test_decode(self):
        """Test decoding IDs back to string."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        ids = [self.vocab['H'], self.vocab['e'], self.vocab['ll'], self.vocab['o']]
        
        text = decode(ids, id_to_token)
        self.assertEqual(text, "Hello")

    def test_save_and_load(self):
        """Test serialization and deserialization."""
        save(self.vocab, self.merges, self.test_dir)
        
        loaded_vocab, loaded_merges = load(self.test_dir)
        
        self.assertEqual(len(loaded_vocab), 10000)
        self.assertEqual(loaded_vocab['ll'], 8)
        self.assertEqual(loaded_merges, self.merges)


if __name__ == "__main__":
    unittest.main()
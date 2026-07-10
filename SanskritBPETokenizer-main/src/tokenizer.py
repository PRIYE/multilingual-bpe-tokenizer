import json
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import re

class SanskritBPETokenizer:
    def __init__(self, vocab_path:Optional[str] = None  , merges_path: Optional[str] = None, token_path: Optional[str] = None):
        """Initialize the tokenizer with vocabulary and merges"""
        self.vocab = []
        self.merges = {}
        if merges_path:
            self.load_vocab(merges_path)
        if token_path:
            self.load_tokens(token_path)
        if vocab_path:
            self.create_tokens(vocab_path, token_path, merges_path)

    def create_tokens(self, vocab_path, token_path, merges_path):
        dataset = load_dataset(vocab_path)
        text = ''.join([i['translation']['sn'] for i in dataset['train']])
        tokens =  self.regex_sanskrit_tokenize(text)
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
        with open(token_path + '/saved.pkl', 'wb') as f:
            pickle.dump(tokens, f, pickle.HIGHEST_PROTOCOL)
        vocab_size = 5250 # the desired final vocabulary size
        num_merges = vocab_size - 256
        ids = list(tokens) # copy so we don't destroy the original list
        merges = {} # (int, int) -> int
        for i in range(num_merges):
          stats = self.get_stats(ids)
          pair = max(stats, key=stats.get)
          idx = 256 + i
          print(f"merging {pair} into a new token {idx}")
          ids = self.merge(ids, pair, idx)
          merges[pair] = idx
        with open(merges_path + '/merges_saved.pkl', 'wb') as f:
            pickle.dump(merges, f, pickle.HIGHEST_PROTOCOL)   
        print("tokens length:", len(tokens))
        print("ids length:", len(ids))
        print(f"compression ratio: {len(tokens) / len(ids):.2f}X")


    def regex_sanskrit_tokenize(self, text):
        # Basic sandhi patterns
        sandhi_patterns = [
            # # Visarga sandhi
            # r'ः\s*([कखगघङचछजझञटठडढणतथदधनपफबभम])',

            # # Vowel sandhi
            # r'([अआइईउऊऋॠऌॡएऐओऔ])्?\s*([अआइईउऊऋॠऌॡएऐओऔ])',

            # # Consonant sandhi
            # r'([क-ह])्\s*([क-ह])',

            # # Common contractions and combinations
            # r'([क-ह])्([यरलवहमनञणन])',

            # # Anusvara and chandrabindu combinations
            # r'[ंँ]([क-ह])',

            # # Handle special cases like ज्ञ, क्ष
            # r'(ज्ञ|क्ष)',

            # # Handle numbers and punctuation
            # r'([०-९])|([।॥,])',
            # # Handle specific compound formations
            # r'([क-ह])्य',  # -ya formations
            # r'([क-ह])्र',  # -ra formations

            # # Handle specific prefixes
            # r'(प्र|उप|अभि|नि|वि|आ|उद्|परि)',

            # # Handle specific suffixes
            # r'(तया|त्वम्|त्वात्)',

            ##################
            # Anusvara and visarga combinations
            r'ं|ः',

            # Common vowel sandhis
            r'ा|ि|ी|ु|ू|ृ|ॄ|ॢ|ॣ|े|ै|ो|ौ',

            # Virama (halant) combinations
            r'्',

            # Common consonant combinations
            r'त्त|त्र|त्व|न्त|न्द|न्ध|श्च|श्व|ष्ट|स्त|स्थ|ह्म|ह्य',

            # Basic word boundaries
            r'\s+',

            # Punctuation and numbers
            r'[।॥॰,!?०-९]+',
        ]

        # Combine all patterns
        pattern = '|'.join(sandhi_patterns)

        # Function to process each match
        def split_token(match):
            token = match.group(0)
            # Add spaces around the matched token
            return f' {token} '

        # Apply the regex
        tokenized_text = re.sub(pattern, split_token, text)
        print('tokenized_text',tokenized_text)

        # Clean up extra spaces and split
        tokens = [token.strip() for token in tokenized_text.split() if token.strip()]

        return ' '.join(tokens)

    def load_tokens(self, token_path: str):
        """Load vocabulary and merges from file"""
        with open(token_path + "/saved.pkl", "rb") as f:
            self.tokens = pickle.load(f)
            print("tokens length:", len(self.tokens))
            chars = sorted(list(set(self.tokens)))


    def load_vocab(self, vocab_path: str):
        """Load vocabulary and merges from file"""
        with open(vocab_path + "/merges_saved.pkl", "rb") as f:
            self.merges = pickle.load(f)
            #print(self.merges)
            # Create reverse vocab from merges
            self.vocab = {idx: bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in self.merges.items():
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            #print(self.vocab)

    def get_stats(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """Count frequency of token pairs"""
        stats = {}
        for pair in zip(tokens, tokens[1:]): # Pythonic way to iterate consecutive elements
            stats[pair] = stats.get(pair, 0) + 1
        return stats

    def merge(self, tokens: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merge all occurrences of a token pair"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text 
    
if __name__ == "__main__":
    # Create tokens from text
    vocab_path = 'rahular/itihasa' # loading sansakrit text from huggingface
    #SanskritBPETokenizer(vocab_path = vocab_path, merges_path='/Users/priye/Desktop/ERAV3/SanskritBPETokenizer' ,  token_path='/Users/priye/Desktop/ERAV3/SanskritBPETokenizer' )

    # Example usage
    tokenizer = SanskritBPETokenizer(merges_path='/Users/priye/Desktop/ERAV3/SanskritBPETokenizer/data/vocab' ,  token_path='/Users/priye/Desktop/ERAV3/SanskritBPETokenizer/data/vocab' )
    
    sample_text = "विश्वामित्रवचः श्रुत्वा राघवः सहलक्ष्मणः। विस्मयं परमं गत्वा विश्वामित्रमथाब्रवीत्॥"
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original text: {sample_text}")
    print(f"Encoded tokens: {encoded}")
    print(f"Decoded text: {decoded}")
    print(tokenizer.decode(tokenizer.encode(sample_text)))
    assert sample_text == tokenizer.decode(tokenizer.encode(sample_text))

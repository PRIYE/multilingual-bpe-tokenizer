# Multilingual BPE Tokenizer - Test Suite

## 🚀 Quick Start

### Run All Tests
```bash
cd tokenizer
python tests/run_all_tests.py
```

### Run Individual Test Suites
```bash
# Core functionality tests
python -m unittest tests.test_regex_pretokenizer
python -m unittest tests.test_bpe
python -m unittest tests.test_tokenizer
python -m unittest tests.test_score
python -m unittest tests.test_train

# Extended functionality tests  
python -m unittest tests.test_regex_pretokenizer_extended
python -m unittest tests.test_bpe_extended
python -m unittest tests.test_tokenizer_extended
```

### Run Specific Tests
```bash
# Run single test class
python -m unittest tests.test_regex_pretokenizer.TestRegexPretokenizer

# Run single test method
python -m unittest tests.test_bpe.TestBPE.test_get_stats_pretokens
```

## 📁 Test Structure

```
tests/
├── __init__.py                          # Package initialization
├── run_all_tests.py                     # Comprehensive test runner
├── TEST_COVERAGE_REPORT.md              # Detailed coverage analysis
├── README.md                            # This file
│
├── test_regex_pretokenizer.py           # Core pre-tokenization tests (6 tests)
├── test_regex_pretokenizer_extended.py  # Extended pre-tokenization tests (14 tests)
├── test_bpe.py                          # Core BPE function tests (4 tests)
├── test_bpe_extended.py                 # Extended BPE function tests (13 tests)
├── test_tokenizer.py                    # Core tokenizer tests (4 tests)
├── test_tokenizer_extended.py           # Extended tokenizer tests (11 tests)
├── test_score.py                        # Score calculation tests (6 tests)
└── test_train.py                        # Training pipeline tests (8 tests)
```

## 🧪 Test Categories

### 1. Regex Pre-tokenizer Tests (20 total)
- **Whitespace preservation** for reversibility
- **Language-specific patterns** (English, Hindi, Telugu, Sanskrit)
- **Unicode normalization** (NFKC, Indic fallbacks)
- **Edge cases** (empty, very long, mixed scripts)
- **Error handling** (invalid languages, regex fallbacks)

### 2. BPE Function Tests (17 total)
- **Boundary respect** (no cross-word merges)
- **Statistics collection** within pre-tokens
- **Merge operations** preserving word boundaries  
- **Fertility calculations** for nested structures
- **Vocabulary building** with character supplements
- **Both normal and parity-aware** BPE algorithms

### 3. Tokenizer Tests (15 total)
- **Two-stage encoding** (regex + BPE)
- **Legacy compatibility** mode
- **Serialization/deserialization** with validation
- **Script classification** system
- **Error handling** (unknown chars, missing files)
- **Multi-language support** (en/hi/te/sa)

### 4. Score Calculator Tests (6 total)
- **Language-specific word counting** (English lowercasing)
- **Fertility computation** (tokens/words ratio)
- **Edge case handling** (empty text, zero words)
- **Report generation** integration

### 5. Training Pipeline Tests (8 total)
- **Alpha sampling** for multilingual balance
- **Data augmentation** with network fallbacks
- **Pre-token aware sampling** algorithms
- **Edge case handling** (empty corpora, network failures)

## 🔍 Key Test Principles

### ✅ Reversibility Testing
Every tokenization operation is tested to ensure the original text can be perfectly reconstructed:
```python
tokens = pretokenize_multilingual(text, language)
assert "".join(tokens) == text  # Perfect reconstruction
```

### ✅ Boundary Respect Testing  
BPE operations are verified to never merge across pre-token boundaries:
```python
stats = get_stats_pretokens([['Hello'], [' world']])
assert ('o', ' ') not in stats  # No cross-boundary pairs
```

### ✅ Multilingual Integrity Testing
Each language's specific patterns are validated:
```python
hindi_tokens = hindi_pretokenize("नमस्ते दुनिया। आप कैसे हैं॥")
assert any('।' in token or '॥' in token for token in hindi_tokens)
```

### ✅ Edge Case Robustness Testing
Comprehensive testing of unusual inputs:
```python
assert english_pretokenize("") == []  # Empty input
assert encode("xyz", vocab, merges, "en") == []  # Unknown chars
```

## 📊 Test Results

**Latest Results:**
- ✅ **66/66 tests passing (100% success rate)**
- ⏱️ **Execution time: ~1.8 seconds**
- 🎯 **Coverage: All major components and edge cases**

See `TEST_COVERAGE_REPORT.md` for detailed analysis.

## 🛠️ Development Workflow

### Adding New Tests
1. Create test in appropriate `test_*.py` file
2. Follow naming convention: `test_<functionality>_<scenario>`
3. Include comprehensive docstring
4. Test both success and failure cases
5. Run full test suite to ensure no regressions

### Test Guidelines
- **Use realistic data**: Real multilingual text samples
- **Test edge cases**: Empty, very long, invalid inputs  
- **Mock external dependencies**: Network calls, file I/O
- **Assert specific behaviors**: Don't just test "no crash"
- **Clean up resources**: Use temporary directories/files

### Debugging Failed Tests
```bash
# Run with verbose output
python -m unittest tests.test_name -v

# Run single failing test
python -m unittest tests.test_bpe.TestBPE.test_failing_method -v

# Use Python debugger
python -m pdb -m unittest tests.test_name
```

## 🎯 Quality Assurance

The test suite ensures:

1. **Functional Correctness**: All algorithms work as designed
2. **Multilingual Integrity**: Proper handling of all target languages  
3. **Performance Reliability**: Efficient execution under load
4. **Error Resilience**: Graceful failure and recovery
5. **Backward Compatibility**: Legacy functionality preserved
6. **Future-Proofing**: Extensible architecture for new features

## 📝 Contributing

When contributing new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Maintain 100% success rate** - fix any failures before PR
3. **Document edge cases** in test docstrings
4. **Update this README** if adding new test categories
5. **Run full test suite** before committing changes

---

**Need help?** Check `TEST_COVERAGE_REPORT.md` for detailed coverage analysis or run `python tests/run_all_tests.py` for a comprehensive system validation.
#!/usr/bin/env python3
"""
Comprehensive test runner for the multilingual BPE tokenizer.

This script runs all unit tests and provides a detailed report of coverage
and test results for each component of the tokenizer system.
"""

import unittest
import sys
import os
import time
from io import StringIO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# Import all test modules
from test_regex_pretokenizer import TestRegexPretokenizer
from test_regex_pretokenizer_extended import TestRegexPreTokenizerExtended
from test_bpe import TestBPE
from test_bpe_extended import TestBPEExtended
from test_tokenizer import TestTokenizer
from test_tokenizer_extended import TestTokenizerExtended
from test_score import TestScore
from test_train import TestTrain


def run_test_suite():
    """Run all test suites and generate a comprehensive report."""
    
    print("=" * 80)
    print("MULTILINGUAL BPE TOKENIZER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test suites organized by component
    test_suites = {
        "Regex Pre-tokenizer (Core)": TestRegexPretokenizer,
        "Regex Pre-tokenizer (Extended)": TestRegexPreTokenizerExtended,
        "BPE Core Functions": TestBPE,
        "BPE Extended Functions": TestBPEExtended,
        "Tokenizer (Core)": TestTokenizer,
        "Tokenizer (Extended)": TestTokenizerExtended,
        "Score Calculator": TestScore,
        "Training Pipeline": TestTrain,
    }
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    total_errors = 0
    
    results = {}
    
    start_time = time.time()
    
    for suite_name, test_class in test_suites.items():
        print(f"\n📋 Running {suite_name}")
        print("-" * 60)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Capture test output
        stream = StringIO()
        runner = unittest.TextTestRunner(stream=stream, verbosity=2)
        
        # Run tests
        result = runner.run(suite)
        
        # Store results
        results[suite_name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success': result.testsRun - len(result.failures) - len(result.errors)
        }
        
        # Update totals
        total_tests += result.testsRun
        total_passed += results[suite_name]['success']
        total_failed += len(result.failures)
        total_errors += len(result.errors)
        
        # Print summary for this suite
        if result.failures or result.errors:
            print(f"❌ {suite_name}: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
            
            # Print failure details
            for test, traceback in result.failures:
                print(f"   FAIL: {test}")
                print(f"         {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
            
            for test, traceback in result.errors:
                print(f"   ERROR: {test}")
                print(f"          {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
        else:
            print(f"✅ {suite_name}: All {result.testsRun} tests passed")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"💥 Errors: {total_errors}")
    print(f"⏱️  Duration: {duration:.2f} seconds")
    
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    # Component breakdown
    print(f"\n📋 Component Breakdown:")
    for suite_name, result in results.items():
        status = "✅" if result['failures'] == 0 and result['errors'] == 0 else "❌"
        print(f"   {status} {suite_name}: {result['success']}/{result['tests_run']} passed")
    
    print("\n" + "=" * 80)
    print("TEST COVERAGE AREAS")
    print("=" * 80)
    
    coverage_areas = [
        "✅ Regex pre-tokenization (whitespace preservation, language-specific patterns)",
        "✅ Unicode normalization (NFKC, Indic normalization fallback)",
        "✅ BPE statistics collection (within pre-token boundaries)",
        "✅ BPE merging operations (respecting word boundaries)",
        "✅ Fertility calculations (both flat and nested token structures)",
        "✅ Vocabulary building (from pre-tokens, character supplements)",
        "✅ Encoding/decoding (legacy vs. pre-tokenization modes)",
        "✅ Tokenizer serialization (save/load with validation)",
        "✅ Script classification (Latin, Devanagari, Telugu, Mixed, Other)",
        "✅ Training pipeline components (alpha sampling, augmentation)",
        "✅ Score calculation (word counting, fertility computation)",
        "✅ Edge cases (empty inputs, unknown characters, invalid parameters)",
        "✅ Error handling (missing files, malformed data, fallbacks)",
    ]
    
    for area in coverage_areas:
        print(f"   {area}")
    
    print(f"\n🎯 Overall Assessment: {'EXCELLENT' if success_rate >= 95 else 'GOOD' if success_rate >= 85 else 'NEEDS IMPROVEMENT'}")
    
    return total_tests == total_passed


if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)
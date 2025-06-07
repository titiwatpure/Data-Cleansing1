"""
Test suite for Data Cleansing Pipeline
ชุดทดสอบสำหรับระบบทำความสะอาดข้อมูล

This module contains comprehensive tests for all components
of the data cleansing pipeline system.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from .test_data_loader import TestDataLoader
from .test_data_cleaner import TestDataCleaner
from .test_data_transformer import TestDataTransformer
from .test_data_validator import TestDataValidator
from .test_reporter import TestReporter
from .test_utils import TestUtils


def create_test_suite():
    """
    สร้างชุดทดสอบที่รวมทุกการทดสอบ
    Create a comprehensive test suite combining all tests.
    """
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataLoader,
        TestDataCleaner,
        TestDataTransformer,
        TestDataValidator,
        TestReporter,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


def run_all_tests():
    """
    รันการทดสอบทั้งหมด
    Run all tests with detailed output.
    """
    print("🧪 เริ่มการทดสอบระบบทำความสะอาดข้อมูล")
    print("🧪 Starting Data Cleansing Pipeline Tests")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        failfast=False
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 สรุปผลการทดสอบ / Test Summary:")
    print(f"✅ ทดสอบผ่าน / Tests Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ ทดสอบล้มเหลว / Tests Failed: {len(result.failures)}")
    print(f"💥 ข้อผิดพลาด / Errors: {len(result.errors)}")
    print(f"📈 อัตราความสำเร็จ / Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


def run_specific_test(test_name):
    """
    รันการทดสอบเฉพาะ
    Run a specific test module.
    
    Args:
        test_name (str): Name of the test module to run
    """
    test_mapping = {
        'loader': TestDataLoader,
        'cleaner': TestDataCleaner,
        'transformer': TestDataTransformer,
        'validator': TestDataValidator,
        'reporter': TestReporter,
        'utils': TestUtils
    }
    
    if test_name not in test_mapping:
        print(f"❌ ไม่พบการทดสอบ '{test_name}'")
        print(f"❌ Test '{test_name}' not found")
        print(f"✅ การทดสอบที่มี / Available tests: {', '.join(test_mapping.keys())}")
        return False
    
    print(f"🧪 รันการทดสอบ {test_name}")
    print(f"🧪 Running {test_name} tests")
    print("-" * 40)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(test_mapping[test_name])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data Cleansing Pipeline Test Runner"
    )
    parser.add_argument(
        '--test', '-t',
        help="Run specific test (loader, cleaner, transformer, validator, reporter, utils)",
        default=None
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help="Run all tests"
    )
    
    args = parser.parse_args()
    
    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()
    
    if success:
        print("\n🎉 การทดสอบสำเร็จทั้งหมด!")
        print("🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n💥 การทดสอบล้มเหลว!")
        print("💥 Some tests failed!")
        sys.exit(1)

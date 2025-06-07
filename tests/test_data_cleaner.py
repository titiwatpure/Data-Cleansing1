"""
Unit tests for data_cleaner module
การทดสอบสำหรับโมดูลทำความสะอาดข้อมูล
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_cleaner import DataCleaner


class TestDataCleaner(unittest.TestCase):
    """ทดสอบการทำงานของ DataCleaner"""
    
    def setUp(self):
        """ตั้งค่าเริ่มต้นสำหรับการทดสอบ"""
        self.cleaner = DataCleaner()
        
        # Create sample data with various issues
        self.messy_data = pd.DataFrame({
            'id': [1, 2, 2, 4, 5, 6],  # Duplicate ID
            'name': ['Alice', 'Bob', 'Bob', np.nan, '  Eve  ', 'Frank'],  # Missing and whitespace
            'age': [25, 30, 30, 35, -5, 150],  # Duplicate and outliers
            'email': ['alice@test.com', 'invalid_email', 'bob@test.com', np.nan, 'eve@test.com', 'frank@test.com'],
            'salary': [50000, np.nan, 60000, 70000, 80000, 999999],  # Missing and outlier
            'date': ['2023-01-01', '2023-02-01', '2023-02-01', '2023-03-01', 'invalid_date', '2023-05-01']
        })
        
        # Clean data for comparison
        self.clean_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
    
    def test_preprocess_data_basic(self):
        """ทดสอบการประมวลผลเบื้องต้น"""
        result = self.cleaner.preprocess(self.messy_data.copy())
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreaterEqual(len(result), 1)  # Should have at least some data
    
    def test_handle_missing_data_drop(self):
        """ทดสอบการจัดการข้อมูลที่หายไปด้วยการลบ"""
        data_with_nulls = self.messy_data.copy()
        
        result = self.cleaner.handle_missing_data(data_with_nulls, strategy='drop')
        
        # Should have fewer rows due to dropped nulls
        self.assertLess(len(result), len(data_with_nulls))
        self.assertEqual(result.isnull().sum().sum(), 0)  # No nulls remaining
    
    def test_handle_missing_data_fill_mean(self):
        """ทดสอบการจัดการข้อมูลที่หายไปด้วยค่าเฉลี่ย"""
        data_with_nulls = pd.DataFrame({
            'numbers': [1, 2, np.nan, 4, 5],
            'text': ['A', 'B', np.nan, 'D', 'E']
        })
        
        result = self.cleaner.handle_missing_data(data_with_nulls, strategy='fill_mean')
        
        # Check that numeric nulls are filled with mean
        self.assertFalse(result['numbers'].isnull().any())
        self.assertEqual(result['numbers'].iloc[2], 3.0)  # Mean of [1,2,4,5] = 3
    
    def test_handle_missing_data_fill_mode(self):
        """ทดสอบการจัดการข้อมูลที่หายไปด้วยค่าที่พบบ่อยที่สุด"""
        data_with_nulls = pd.DataFrame({
            'category': ['A', 'B', 'A', np.nan, 'A', 'B']
        })
        
        result = self.cleaner.handle_missing_data(data_with_nulls, strategy='fill_mode')
        
        # Check that null is filled with mode (A appears 3 times)
        self.assertFalse(result['category'].isnull().any())
        self.assertEqual(result['category'].iloc[3], 'A')
    
    def test_remove_duplicates_keep_first(self):
        """ทดสอบการลบข้อมูลซ้ำโดยเก็บแถวแรก"""
        data_with_dups = pd.DataFrame({
            'id': [1, 2, 2, 3],
            'name': ['A', 'B', 'B', 'C'],
            'value': [10, 20, 25, 30]
        })
        
        result = self.cleaner.remove_duplicates(data_with_dups, subset=['id', 'name'], keep='first')
        
        self.assertEqual(len(result), 3)  # One duplicate removed
        self.assertEqual(result[result['id'] == 2]['value'].iloc[0], 20)  # First occurrence kept
    
    def test_remove_duplicates_keep_last(self):
        """ทดสอบการลบข้อมูลซ้ำโดยเก็บแถวสุดท้าย"""
        data_with_dups = pd.DataFrame({
            'id': [1, 2, 2, 3],
            'name': ['A', 'B', 'B', 'C'],
            'value': [10, 20, 25, 30]
        })
        
        result = self.cleaner.remove_duplicates(data_with_dups, subset=['id', 'name'], keep='last')
        
        self.assertEqual(len(result), 3)  # One duplicate removed
        self.assertEqual(result[result['id'] == 2]['value'].iloc[0], 25)  # Last occurrence kept
    
    def test_standardize_text_basic(self):
        """ทดสอบการปรับมาตรฐานข้อความพื้นฐาน"""
        data_with_text = pd.DataFrame({
            'text': ['  Hello World  ', 'HELLO WORLD', 'hello world', '  HELLO  WORLD  ']
        })
        
        result = self.cleaner.standardize_text(data_with_text, 'text')
        
        # Check that all text is standardized
        for text in result['text']:
            self.assertEqual(text.strip(), text)  # No leading/trailing whitespace
            self.assertNotIn('  ', text)  # No multiple spaces
    
    def test_standardize_formats_email(self):
        """ทดสอบการปรับมาตรฐานรูปแบบอีเมล"""
        data_with_emails = pd.DataFrame({
            'email': ['Alice@Test.COM', '  bob@test.com  ', 'CHARLIE@TEST.COM']
        })
        
        result = self.cleaner.standardize_formats(data_with_emails, 'email', 'email')
        
        # All emails should be lowercase and trimmed
        for email in result['email']:
            self.assertEqual(email, email.lower().strip())
    
    def test_detect_outliers_iqr(self):
        """ทดสอบการตรวจจับค่าผิดปกติด้วยวิธี IQR"""
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10]  # 100 is an outlier
        })
        
        outliers = self.cleaner.detect_outliers(data_with_outliers, 'values', method='iqr')
        
        self.assertIn(5, outliers)  # Index 5 contains value 100
        self.assertEqual(len(outliers), 1)
    
    def test_detect_outliers_zscore(self):
        """ทดสอบการตรวจจับค่าผิดปกติด้วยวิธี Z-Score"""
        data_with_outliers = pd.DataFrame({
            'values': [1, 2, 3, 4, 5, 100, 7, 8, 9, 10]  # 100 is an outlier
        })
        
        outliers = self.cleaner.detect_outliers(data_with_outliers, 'values', method='zscore', threshold=2)
        
        self.assertGreater(len(outliers), 0)  # Should detect at least one outlier
        self.assertIn(5, outliers)  # Index 5 contains value 100
    
    def test_validate_data_consistency_email(self):
        """ทดสอบการตรวจสอบความสอดคล้องของข้อมูลอีเมล"""
        data_with_invalid = pd.DataFrame({
            'email': ['valid@email.com', 'invalid_email', 'another@valid.com', 'also_invalid']
        })
        
        issues = self.cleaner.validate_data_consistency(data_with_invalid, 'email', 'email')
        
        self.assertEqual(len(issues), 2)  # Two invalid emails
        self.assertIn(1, issues)  # Index 1: invalid_email
        self.assertIn(3, issues)  # Index 3: also_invalid
    
    def test_validate_data_consistency_phone(self):
        """ทดสอบการตรวจสอบความสอดคล้องของข้อมูลเบอร์โทร"""
        data_with_phones = pd.DataFrame({
            'phone': ['0812345678', '+66812345678', '081-234-5678', 'invalid_phone', '06123456789']
        })
        
        issues = self.cleaner.validate_data_consistency(data_with_phones, 'phone', 'phone')
        
        self.assertGreater(len(issues), 0)  # Should find invalid phones
    
    def test_clean_data_full_pipeline(self):
        """ทดสอบการทำความสะอาดข้อมูลแบบครบวงจร"""
        messy_data = self.messy_data.copy()
        
        result = self.cleaner.clean_data(messy_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('cleaned_data', result)
        self.assertIn('log', result)
        
        cleaned_df = result['cleaned_data']
        log = result['log']
        
        # Check that data is cleaned
        self.assertIsInstance(cleaned_df, pd.DataFrame)
        self.assertGreater(len(log), 0)  # Should have some log entries
        
        # Check that some cleaning was performed
        self.assertLessEqual(len(cleaned_df), len(messy_data))  # Duplicates or invalid rows removed
    
    def test_get_cleaning_summary(self):
        """ทดสอบการสร้างสรุปการทำความสะอาด"""
        messy_data = self.messy_data.copy()
        
        # Clean data first
        clean_result = self.cleaner.clean_data(messy_data)
        
        summary = self.cleaner.get_cleaning_summary(
            messy_data, 
            clean_result['cleaned_data'], 
            clean_result['log']
        )
        
        self.assertIsInstance(summary, dict)
        self.assertIn('original_rows', summary)
        self.assertIn('cleaned_rows', summary)
        self.assertIn('removed_rows', summary)
        self.assertIn('actions_performed', summary)


if __name__ == '__main__':
    unittest.main()

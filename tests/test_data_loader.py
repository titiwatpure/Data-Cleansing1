"""
Unit tests for data_loader module
การทดสอบสำหรับโมดูลโหลดข้อมูล
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """ทดสอบการทำงานของ DataLoader"""
    
    def setUp(self):
        """ตั้งค่าเริ่มต้นสำหรับการทดสอบ"""
        self.loader = DataLoader()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
    
    def tearDown(self):
        """ทำความสะอาดหลังการทดสอบ"""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_csv_valid_file(self):
        """ทดสอบการโหลดไฟล์ CSV ที่ถูกต้อง"""
        # Create test CSV file
        csv_path = os.path.join(self.temp_dir, 'test.csv')
        self.sample_data.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Load data
        result = self.loader.load_csv(csv_path)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result.columns), ['id', 'name', 'age', 'email', 'salary'])
        self.assertEqual(result['name'].iloc[0], 'Alice')
    
    def test_load_csv_nonexistent_file(self):
        """ทดสอบการโหลดไฟล์ CSV ที่ไม่มีอยู่"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.csv')
        
        result = self.loader.load_csv(nonexistent_path)
        
        self.assertIsNone(result)
    
    def test_load_excel_valid_file(self):
        """ทดสอบการโหลดไฟล์ Excel ที่ถูกต้อง"""
        # Create test Excel file
        excel_path = os.path.join(self.temp_dir, 'test.xlsx')
        self.sample_data.to_excel(excel_path, index=False)
        
        # Load data
        result = self.loader.load_excel(excel_path)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['name'].iloc[1], 'Bob')
    
    def test_load_json_valid_file(self):
        """ทดสอบการโหลดไฟล์ JSON ที่ถูกต้อง"""
        # Create test JSON file
        json_path = os.path.join(self.temp_dir, 'test.json')
        self.sample_data.to_json(json_path, orient='records', indent=2)
        
        # Load data
        result = self.loader.load_json(json_path)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertIn('name', result.columns)
    
    def test_load_json_invalid_format(self):
        """ทดสอบการโหลดไฟล์ JSON ที่มีรูปแบบไม่ถูกต้อง"""
        # Create invalid JSON file
        json_path = os.path.join(self.temp_dir, 'invalid.json')
        with open(json_path, 'w') as f:
            f.write('{"invalid": json format}')
        
        result = self.loader.load_json(json_path)
        
        self.assertIsNone(result)
    
    def test_detect_encoding_utf8(self):
        """ทดสอบการตรวจจับ encoding UTF-8"""
        # Create UTF-8 file
        csv_path = os.path.join(self.temp_dir, 'utf8.csv')
        test_data = pd.DataFrame({
            'text': ['สวัสดี', 'Hello', '你好', 'Bonjour']
        })
        test_data.to_csv(csv_path, index=False, encoding='utf-8')
        
        encoding = self.loader.detect_encoding(csv_path)
        
        self.assertIn('utf', encoding.lower())
    
    def test_load_data_auto_format_csv(self):
        """ทดสอบการโหลดข้อมูลอัตโนมัติสำหรับ CSV"""
        csv_path = os.path.join(self.temp_dir, 'auto_test.csv')
        self.sample_data.to_csv(csv_path, index=False)
        
        result = self.loader.load_data(csv_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
    
    def test_load_data_auto_format_excel(self):
        """ทดสอบการโหลดข้อมูลอัตโนมัติสำหรับ Excel"""
        excel_path = os.path.join(self.temp_dir, 'auto_test.xlsx')
        self.sample_data.to_excel(excel_path, index=False)
        
        result = self.loader.load_data(excel_path)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
    
    def test_load_data_unsupported_format(self):
        """ทดสอบการโหลดไฟล์รูปแบบที่ไม่รองรับ"""
        unsupported_path = os.path.join(self.temp_dir, 'test.txt')
        with open(unsupported_path, 'w') as f:
            f.write('Some text content')
        
        result = self.loader.load_data(unsupported_path)
        
        self.assertIsNone(result)
    
    def test_get_data_info(self):
        """ทดสอบการดึงข้อมูลสถิติของ DataFrame"""
        info = self.loader.get_data_info(self.sample_data)
        
        self.assertIsInstance(info, dict)
        self.assertEqual(info['rows'], 5)
        self.assertEqual(info['columns'], 5)
        self.assertIn('memory_usage', info)
        self.assertIn('column_types', info)
        self.assertIn('missing_values', info)
    
    def test_preview_data(self):
        """ทดสอบการแสดงตัวอย่างข้อมูล"""
        preview = self.loader.preview_data(self.sample_data, n_rows=3)
        
        self.assertIsInstance(preview, pd.DataFrame)
        self.assertEqual(len(preview), 3)
        self.assertEqual(list(preview.columns), list(self.sample_data.columns))
    
    def test_preview_data_more_than_available(self):
        """ทดสอบการแสดงตัวอย่างข้อมูลเกินจำนวนที่มี"""
        preview = self.loader.preview_data(self.sample_data, n_rows=10)
        
        self.assertEqual(len(preview), 5)  # Should return all available rows


if __name__ == '__main__':
    unittest.main()

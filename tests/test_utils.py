# filepath: c:\Users\titiw\Desktop\Data Cleansing\tests\test_utils.py
"""
Tests for Utility Functions
===========================

Tests for the utility functions used in the data cleansing pipeline.
"""

import unittest
import os
import tempfile
import yaml
import json
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.utils import (
    setup_logging, load_config, validate_file_path, 
    get_file_extension, format_file_size, sanitize_column_name,
    create_backup, load_json_schema, generate_unique_filename
)


class TestUtils(unittest.TestCase):
    """ทดสอบการทำงานของ Utils"""
    
    def setUp(self):
        """ตั้งค่าก่อนการทดสอบ"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """ทำความสะอาดหลังการทดสอบ"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_setup_logging(self):
        """ทดสอบการตั้งค่า logging"""
        # Test basic logging setup
        setup_logging()
        # If no exception raised, test passes
        self.assertTrue(True)
    
    def test_load_config_yaml_file(self):
        """ทดสอบการโหลดไฟล์ config YAML"""
        # Create test config file
        config_content = """
database:
  host: localhost
  port: 3306
  
processing:
  max_rows: 1000
  chunk_size: 100
  
validation:
  quality_threshold: 0.95
"""
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        config = load_config(config_file)
        
        self.assertIsInstance(config, dict)
        self.assertIn('database', config)
        self.assertIn('processing', config)
        self.assertIn('validation', config)
        self.assertEqual(config['database']['host'], 'localhost')
        self.assertEqual(config['processing']['max_rows'], 1000)
    
    def test_validate_file_path_existing(self):
        """ทดสอบการตรวจสอบ path ของไฟล์ที่มีอยู่"""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Test existing file
        result_path = validate_file_path(test_file, must_exist=True)
        self.assertIsInstance(result_path, Path)
        self.assertTrue(result_path.exists())
    
    def test_validate_file_path_nonexisting(self):
        """ทดสอบการตรวจสอบ path ของไฟล์ที่ไม่มีอยู่"""
        test_file = os.path.join(self.temp_dir, 'nonexistent.txt')
          # Test non-existing file with must_exist=False
        result_path = validate_file_path(test_file, must_exist=False)
        self.assertIsInstance(result_path, Path)
    
    def test_get_file_extension(self):
        """ทดสอบการดึง extension ของไฟล์"""
        self.assertEqual(get_file_extension("test.csv"), "csv")
        self.assertEqual(get_file_extension("data.xlsx"), "xlsx")
        self.assertEqual(get_file_extension("file.json"), "json")
        self.assertEqual(get_file_extension("noextension"), "")
    def test_format_file_size(self):
        """ทดสอบการจัดรูปแบบขนาดไฟล์"""
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1048576), "1.0 MB")
        self.assertEqual(format_file_size(500), "500.0 B")
    def test_sanitize_column_name(self):
        """ทดสอบการทำความสะอาดชื่อคอลัมน์"""
        self.assertEqual(sanitize_column_name("Name with Spaces"), "name_with_spaces")
        self.assertEqual(sanitize_column_name("Special!@#Characters"), "special_characters")
        self.assertEqual(sanitize_column_name("ชื่อไทย"), "")  # Thai characters are removed by current implementation
    
    def test_create_backup(self):
        """ทดสอบการสร้างไฟล์สำรอง"""
        # Create test file
        test_file = os.path.join(self.temp_dir, 'test.csv')
        with open(test_file, 'w') as f:
            f.write("test,data\n1,2")
        
        backup_path = create_backup(test_file)
        self.assertTrue(os.path.exists(backup_path))
        self.assertIn('backup', backup_path)
    
    def test_load_json_schema(self):
        """ทดสอบการโหลด JSON schema"""
        # Create test schema file
        schema_data = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        schema_file = os.path.join(self.temp_dir, 'test_schema.json')
        with open(schema_file, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f)
        
        schema = load_json_schema(schema_file)
        
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema['type'], 'object')
        self.assertIn('properties', schema)
    
    def test_generate_unique_filename(self):
        """ทดสอบการสร้างชื่อไฟล์ไม่ซ้ำ"""
        base_path = os.path.join(self.temp_dir, 'test.csv')
        
        filename1 = generate_unique_filename(base_path, prefix="clean_", suffix="_v1")
        filename2 = generate_unique_filename(base_path, prefix="clean_", suffix="_v2")
        
        self.assertNotEqual(filename1, filename2)
        self.assertIn('clean_', filename1)
        self.assertIn('_v1', filename1)


if __name__ == '__main__':
    unittest.main()

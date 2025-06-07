"""
Unit tests for data_transformer module
การทดสอบสำหรับโมดูลแปลงข้อมูล
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_transformer import DataTransformer


class TestDataTransformer(unittest.TestCase):
    """ทดสอบการทำงานของ DataTransformer"""
    
    def setUp(self):
        """ตั้งค่าเริ่มต้นสำหรับการทดสอบ"""
        self.transformer = DataTransformer()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
            'last_name': ['Doe', 'Smith', 'Johnson', 'Williams', 'Brown'],
            'age': [25, 30, 35, 40, 45],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT'],
            'join_date': ['2020-01-01', '2019-06-15', '2021-03-10', '2018-12-01', '2022-02-14'],
            'score': [85, 92, 78, 95, 88]
        })
    
    def test_create_feature_basic(self):
        """ทดสอบการสร้างฟีเจอร์พื้นฐาน"""
        data = self.sample_data.copy()
        
        # Create age groups
        def age_group(age):
            if age < 30:
                return 'Young'
            elif age < 40:
                return 'Middle'
            else:
                return 'Senior'
        
        result = self.transformer.create_feature(data, 'age_group', age_group, ['age'])
        
        self.assertIn('age_group', result.columns)
        self.assertEqual(result['age_group'].iloc[0], 'Young')  # Age 25
        self.assertEqual(result['age_group'].iloc[4], 'Senior')  # Age 45
    
    def test_create_feature_multiple_columns(self):
        """ทดสอบการสร้างฟีเจอร์จากหลายคอลัมน์"""
        data = self.sample_data.copy()
        
        # Create full name from first and last name
        def full_name(first, last):
            return f"{first} {last}"
        
        result = self.transformer.create_feature(data, 'full_name', full_name, ['first_name', 'last_name'])
        
        self.assertIn('full_name', result.columns)
        self.assertEqual(result['full_name'].iloc[0], 'John Doe')
        self.assertEqual(result['full_name'].iloc[1], 'Jane Smith')
    
    def test_map_values_dictionary(self):
        """ทดสอบการแมปค่าด้วย dictionary"""
        data = self.sample_data.copy()
        
        dept_mapping = {
            'IT': 'Information Technology',
            'HR': 'Human Resources',
            'Finance': 'Financial Department'
        }
        
        result = self.transformer.map_values(data, 'department', dept_mapping)
        
        self.assertEqual(result['department'].iloc[0], 'Information Technology')
        self.assertEqual(result['department'].iloc[1], 'Human Resources')
        self.assertEqual(result['department'].iloc[3], 'Financial Department')
    
    def test_map_values_function(self):
        """ทดสอบการแมปค่าด้วยฟังก์ชัน"""
        data = self.sample_data.copy()
        
        # Convert age to generation
        def get_generation(age):
            if age < 25:
                return 'Gen Z'
            elif age < 40:
                return 'Millennial'
            else:
                return 'Gen X'
        
        result = self.transformer.map_values(data, 'age', get_generation, target_column='generation')
        
        self.assertIn('generation', result.columns)
        self.assertEqual(result['generation'].iloc[0], 'Millennial')  # Age 25
        self.assertEqual(result['generation'].iloc[4], 'Gen X')  # Age 45
    
    def test_split_column_basic(self):
        """ทดสอบการแยกคอลัมน์พื้นฐาน"""
        data = pd.DataFrame({
            'full_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'city_country': ['Bangkok_Thailand', 'London_UK', 'Tokyo_Japan']
        })
        
        result = self.transformer.split_column(data, 'full_name', ' ', ['first', 'last'])
        
        self.assertIn('first', result.columns)
        self.assertIn('last', result.columns)
        self.assertEqual(result['first'].iloc[0], 'John')
        self.assertEqual(result['last'].iloc[0], 'Doe')
    
    def test_split_column_custom_separator(self):
        """ทดสอบการแยกคอลัมน์ด้วยตัวคั่นกำหนดเอง"""
        data = pd.DataFrame({
            'location': ['Bangkok_Thailand', 'London_UK', 'Tokyo_Japan']
        })
        
        result = self.transformer.split_column(data, 'location', '_', ['city', 'country'])
        
        self.assertIn('city', result.columns)
        self.assertIn('country', result.columns)
        self.assertEqual(result['city'].iloc[0], 'Bangkok')
        self.assertEqual(result['country'].iloc[0], 'Thailand')
    
    def test_merge_columns_basic(self):
        """ทดสอบการรวมคอลัมน์พื้นฐาน"""
        data = self.sample_data.copy()
        
        result = self.transformer.merge_columns(
            data, 
            ['first_name', 'last_name'], 
            'full_name', 
            separator=' '
        )
        
        self.assertIn('full_name', result.columns)
        self.assertEqual(result['full_name'].iloc[0], 'John Doe')
        self.assertEqual(result['full_name'].iloc[1], 'Jane Smith')
    
    def test_merge_columns_custom_separator(self):
        """ทดสอบการรวมคอลัมน์ด้วยตัวคั่นกำหนดเอง"""
        data = pd.DataFrame({
            'year': ['2023', '2022', '2021'],
            'month': ['01', '12', '06'],
            'day': ['15', '25', '30']
        })
        
        result = self.transformer.merge_columns(
            data, 
            ['year', 'month', 'day'], 
            'date', 
            separator='-'
        )
        
        self.assertIn('date', result.columns)
        self.assertEqual(result['date'].iloc[0], '2023-01-15')
        self.assertEqual(result['date'].iloc[1], '2022-12-25')
    
    def test_convert_data_types_basic(self):
        """ทดสอบการแปลงประเภทข้อมูลพื้นฐาน"""
        data = pd.DataFrame({
            'id': ['1', '2', '3'],
            'price': ['10.5', '20.75', '30.0'],
            'active': ['True', 'False', 'True']
        })
        
        type_mapping = {
            'id': 'int',
            'price': 'float',
            'active': 'bool'
        }
        
        result = self.transformer.convert_data_types(data, type_mapping)
        
        self.assertEqual(result['id'].dtype, 'int64')
        self.assertEqual(result['price'].dtype, 'float64')
        self.assertEqual(result['active'].dtype, 'bool')
    
    def test_convert_data_types_datetime(self):
        """ทดสอบการแปลงข้อมูลวันที่"""
        data = pd.DataFrame({
            'date_str': ['2023-01-01', '2023-02-15', '2023-03-30'],
            'timestamp': ['2023-01-01 10:30:00', '2023-02-15 14:45:00', '2023-03-30 09:15:00']
        })
        
        type_mapping = {
            'date_str': 'datetime',
            'timestamp': 'datetime'
        }
        
        result = self.transformer.convert_data_types(data, type_mapping)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date_str']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['timestamp']))
    
    def test_normalize_column_minmax(self):
        """ทดสอบการปรับมาตรฐานคอลัมน์ด้วยวิธี Min-Max"""
        data = self.sample_data.copy()
        
        result = self.transformer.normalize_column(data, 'salary', method='minmax')
        
        # Check that values are between 0 and 1
        normalized_values = result['salary_normalized']
        self.assertGreaterEqual(normalized_values.min(), 0)
        self.assertLessEqual(normalized_values.max(), 1)
        self.assertEqual(normalized_values.min(), 0)  # Min should be 0
        self.assertEqual(normalized_values.max(), 1)  # Max should be 1
    
    def test_normalize_column_zscore(self):
        """ทดสอบการปรับมาตรฐานคอลัมน์ด้วยวิธี Z-Score"""
        data = self.sample_data.copy()
        
        result = self.transformer.normalize_column(data, 'score', method='zscore')
        
        # Check that mean is approximately 0 and std is approximately 1
        normalized_values = result['score_normalized']
        self.assertAlmostEqual(normalized_values.mean(), 0, places=10)
        self.assertAlmostEqual(normalized_values.std(), 1, places=10)
    
    def test_bin_numeric_data_equal_width(self):
        """ทดสอบการแบ่งกลุ่มข้อมูลตัวเลขด้วยความกว้างเท่ากัน"""
        data = self.sample_data.copy()
        
        result = self.transformer.bin_numeric_data(
            data, 
            'age', 
            bins=3, 
            method='equal_width',
            labels=['Young', 'Middle', 'Senior']
        )
        
        self.assertIn('age_binned', result.columns)
        binned_values = result['age_binned'].unique()
        self.assertLessEqual(len(binned_values), 3)  # Should have at most 3 bins
    
    def test_bin_numeric_data_quantile(self):
        """ทดสอบการแบ่งกลุ่มข้อมูลตัวเลขด้วยควอนไทล์"""
        data = self.sample_data.copy()
        
        result = self.transformer.bin_numeric_data(
            data, 
            'salary', 
            bins=4, 
            method='quantile'
        )
        
        self.assertIn('salary_binned', result.columns)
        # Each bin should have roughly equal number of observations
        value_counts = result['salary_binned'].value_counts()
        self.assertGreater(len(value_counts), 0)
    
    def test_encode_categorical_onehot(self):
        """ทดสอบการเข้ารหัสข้อมูลหมวดหมู่ด้วย One-Hot"""
        data = self.sample_data.copy()
        
        result = self.transformer.encode_categorical(data, 'department', method='onehot')
        
        # Should create new columns for each category
        expected_columns = ['department_Finance', 'department_HR', 'department_IT']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that values are binary (0 or 1)
        for col in expected_columns:
            unique_values = result[col].unique()
            self.assertTrue(all(val in [0, 1] for val in unique_values))
    
    def test_encode_categorical_label(self):
        """ทดสอบการเข้ารหัสข้อมูลหมวดหมู่ด้วย Label Encoding"""
        data = self.sample_data.copy()
        
        result = self.transformer.encode_categorical(data, 'department', method='label')
        
        self.assertIn('department_encoded', result.columns)
        
        # Values should be numeric
        encoded_values = result['department_encoded']
        self.assertTrue(pd.api.types.is_numeric_dtype(encoded_values))
        
        # Should have integer values
        self.assertTrue(all(isinstance(val, (int, np.integer)) for val in encoded_values))
    
    def test_transform_data_pipeline(self):
        """ทดสอบการแปลงข้อมูลแบบครบวงจร"""
        data = self.sample_data.copy()
        
        # Define transformations
        transformations = [
            {
                'type': 'create_feature',
                'name': 'salary_category',
                'function': lambda x: 'High' if x > 70000 else 'Low',
                'columns': ['salary']
            },
            {
                'type': 'normalize',
                'column': 'age',
                'method': 'minmax'
            }
        ]
        
        result = self.transformer.transform_data(data, transformations)
        
        self.assertIsInstance(result, dict)
        self.assertIn('transformed_data', result)
        self.assertIn('log', result)
        
        transformed_df = result['transformed_data']
        self.assertIn('salary_category', transformed_df.columns)
        self.assertIn('age_normalized', transformed_df.columns)


if __name__ == '__main__':
    unittest.main()

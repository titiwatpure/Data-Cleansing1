"""
Unit tests for reporter module
การทดสอบสำหรับโมดูลรายงาน
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.reporter import Reporter


class TestReporter(unittest.TestCase):
    """ทดสอบการทำงานของ Reporter"""
    
    def setUp(self):
        """ตั้งค่าเริ่มต้นสำหรับการทดสอบ"""
        self.reporter = Reporter()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 'david@test.com', 'eve@test.com'],
            'salary': [50000, 60000, 70000, 80000, 90000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
        })
        
        # Sample validation results
        self.validation_results = {
            'quality_score': 0.85,
            'metrics': {
                'completeness': 0.90,
                'uniqueness': 0.95,
                'consistency': 0.80,
                'accuracy': 0.88,
                'validity': 0.92
            },
            'issues': [
                'Missing values found in column: name',
                'Invalid email format in column: email',
                'Duplicate values found in column: id'
            ],
            'recommendations': [
                'Consider filling missing values in name column',
                'Validate and correct email formats',
                'Remove or investigate duplicate ID values'
            ]
        }
        
        # Sample cleaning log
        self.cleaning_log = [
            'Removed 2 duplicate rows',
            'Filled 3 missing values in name column',
            'Standardized email formats',
            'Detected 1 outlier in salary column'
        ]
    
    def tearDown(self):
        """ทำความสะอาดหลังการทดสอบ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_generate_data_summary_basic(self):
        """ทดสอบการสร้างสรุปข้อมูลพื้นฐาน"""
        summary = self.reporter.generate_data_summary(self.sample_data)
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_rows'], 5)
        self.assertEqual(summary['total_columns'], 6)
        self.assertIn('column_info', summary)
        self.assertIn('missing_values', summary)
        self.assertIn('data_types', summary)
    
    def test_generate_data_summary_with_nulls(self):
        """ทดสอบการสร้างสรุปข้อมูลที่มีค่า null"""
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[2, 'name'] = np.nan
        data_with_nulls.loc[4, 'email'] = np.nan
        
        summary = self.reporter.generate_data_summary(data_with_nulls)
        
        self.assertEqual(summary['missing_values']['name'], 1)
        self.assertEqual(summary['missing_values']['email'], 1)
        self.assertEqual(summary['missing_values']['age'], 0)
    
    def test_generate_quality_report_basic(self):
        """ทดสอบการสร้างรายงานคุณภาพพื้นฐาน"""
        report = self.reporter.generate_quality_report(self.validation_results)
        
        self.assertIsInstance(report, dict)
        self.assertIn('overall_score', report)
        self.assertIn('grade', report)
        self.assertIn('metric_details', report)
        self.assertIn('issues_summary', report)
        self.assertIn('recommendations', report)
        
        self.assertEqual(report['overall_score'], 0.85)
    
    def test_generate_quality_report_grading(self):
        """ทดสอบการให้เกรดคุณภาพข้อมูล"""
        # Test excellent grade
        excellent_results = {'quality_score': 0.95, 'metrics': {}, 'issues': [], 'recommendations': []}
        report = self.reporter.generate_quality_report(excellent_results)
        self.assertEqual(report['grade'], 'ยอดเยี่ยม (Excellent)')
        
        # Test poor grade
        poor_results = {'quality_score': 0.45, 'metrics': {}, 'issues': [], 'recommendations': []}
        report = self.reporter.generate_quality_report(poor_results)
        self.assertEqual(report['grade'], 'ต้องปรับปรุง (Needs Improvement)')
    
    def test_generate_cleaning_report_basic(self):
        """ทดสอบการสร้างรายงานการทำความสะอาด"""
        original_data = self.sample_data.copy()
        # Simulate cleaned data (remove one row)
        cleaned_data = self.sample_data.iloc[:-1].copy()
        
        report = self.reporter.generate_cleaning_report(
            original_data, 
            cleaned_data, 
            self.cleaning_log
        )
        
        self.assertIsInstance(report, dict)
        self.assertIn('summary', report)
        self.assertIn('actions_taken', report)
        self.assertIn('data_changes', report)
        
        # Check data changes
        self.assertEqual(report['data_changes']['rows_before'], 5)
        self.assertEqual(report['data_changes']['rows_after'], 4)
        self.assertEqual(report['data_changes']['rows_removed'], 1)
    
    def test_create_html_report_basic(self):
        """ทดสอบการสร้างรายงาน HTML พื้นฐาน"""
        html_content = self.reporter.create_html_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log
        )
        
        self.assertIsInstance(html_content, str)
        self.assertIn('<html>', html_content)
        self.assertIn('<head>', html_content)
        self.assertIn('<body>', html_content)
        self.assertIn('Data Quality Report', html_content)
        self.assertIn('Quality Score', html_content)
    
    def test_create_html_report_with_thai_content(self):
        """ทดสอบการสร้างรายงาน HTML ที่มีเนื้อหาภาษาไทย"""
        html_content = self.reporter.create_html_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log
        )
        
        # Check for Thai content
        self.assertIn('รายงานคุณภาพข้อมูล', html_content)
        self.assertIn('คะแนนคุณภาพ', html_content)
        self.assertIn('สรุปข้อมูล', html_content)
    
    def test_save_report_html(self):
        """ทดสอบการบันทึกรายงาน HTML"""
        output_path = os.path.join(self.temp_dir, 'test_report.html')
        
        success = self.reporter.save_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log,
            output_path,
            format='html'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('<html>', content)
            self.assertIn('Data Quality Report', content)
    
    def test_save_report_json(self):
        """ทดสอบการบันทึกรายงาน JSON"""
        output_path = os.path.join(self.temp_dir, 'test_report.json')
        
        success = self.reporter.save_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log,
            output_path,
            format='json'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content
        import json
        with open(output_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            self.assertIn('data_summary', content)
            self.assertIn('quality_report', content)
            self.assertIn('cleaning_report', content)
    
    def test_save_report_txt(self):
        """ทดสอบการบันทึกรายงานข้อความ"""
        output_path = os.path.join(self.temp_dir, 'test_report.txt')
        
        success = self.reporter.save_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log,
            output_path,
            format='txt'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Check file content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('DATA QUALITY REPORT', content)
            self.assertIn('Quality Score', content)
    
    def test_save_report_unsupported_format(self):
        """ทดสอบการบันทึกรายงานในรูปแบบที่ไม่รองรับ"""
        output_path = os.path.join(self.temp_dir, 'test_report.pdf')
        
        success = self.reporter.save_report(
            self.sample_data,
            self.validation_results,
            self.cleaning_log,
            output_path,
            format='pdf'  # Not implemented yet
        )
        
        self.assertFalse(success)
        self.assertFalse(os.path.exists(output_path))
    
    def test_generate_executive_summary(self):
        """ทดสอบการสร้างสรุปผู้บริหาร"""
        summary = self.reporter.generate_executive_summary(
            self.sample_data,
            self.validation_results,
            self.cleaning_log
        )
        
        self.assertIsInstance(summary, dict)
        self.assertIn('data_overview', summary)
        self.assertIn('quality_assessment', summary)
        self.assertIn('key_findings', summary)
        self.assertIn('recommendations', summary)
        self.assertIn('actions_taken', summary)
    
    def test_create_data_profile(self):
        """ทดสอบการสร้างโปรไฟล์ข้อมูล"""
        profile = self.reporter.create_data_profile(self.sample_data)
        
        self.assertIsInstance(profile, dict)
        self.assertIn('basic_info', profile)
        self.assertIn('column_profiles', profile)
        
        # Check column profiles
        for column in self.sample_data.columns:
            self.assertIn(column, profile['column_profiles'])
            col_profile = profile['column_profiles'][column]
            self.assertIn('data_type', col_profile)
            self.assertIn('non_null_count', col_profile)
            self.assertIn('unique_count', col_profile)
    
    def test_format_thai_numbers(self):
        """ทดสอบการจัดรูปแบบตัวเลขภาษาไทย"""
        # Test if the method exists and works
        if hasattr(self.reporter, '_format_thai_numbers'):
            formatted = self.reporter._format_thai_numbers(1234567.89)
            self.assertIsInstance(formatted, str)
            self.assertIn(',', formatted)  # Should have thousand separators
    
    def test_get_quality_interpretation_excellent(self):
        """ทดสอบการแปลความหมายคุณภาพระดับยอดเยี่ยม"""
        if hasattr(self.reporter, '_get_quality_interpretation'):
            interpretation = self.reporter._get_quality_interpretation(0.95)
            self.assertIn('ยอดเยี่ยม', interpretation)
    
    def test_get_quality_interpretation_poor(self):
        """ทดสอบการแปลความหมายคุณภาพระดับต่ำ"""
        if hasattr(self.reporter, '_get_quality_interpretation'):
            interpretation = self.reporter._get_quality_interpretation(0.40)
            self.assertIn('ต้องปรับปรุง', interpretation)


if __name__ == '__main__':
    unittest.main()

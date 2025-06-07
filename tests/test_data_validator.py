"""
Unit tests for data_validator module
การทดสอบสำหรับโมดูลตรวจสอบข้อมูล
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.data_validator import DataValidator


class TestDataValidator(unittest.TestCase):
    """ทดสอบการทำงานของ DataValidator"""
    
    def setUp(self):
        """ตั้งค่าเริ่มต้นสำหรับการทดสอบ"""
        self.validator = DataValidator()
        
        # Create sample data for testing
        self.test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
            'email': ['alice@test.com', 'bob@test.com', 'invalid_email', 'david@test.com', 'eve@test.com', 
                     'frank@test.com', 'grace@test.com', 'henry@test.com', 'ivy@test.com', 'jack@test.com'],
            'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            'salary': [50000, 60000, np.nan, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
            'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance'],
            'join_date': ['2020-01-01', '2019-06-15', '2021-03-10', '2018-12-01', '2022-02-14',
                         '2020-08-20', '2019-11-05', '2021-07-12', '2020-03-25', '2022-01-30']
        })
        
        # Data with quality issues
        self.poor_quality_data = pd.DataFrame({
            'id': [1, 1, 3, 4, 5],  # Duplicate IDs
            'name': [np.nan, 'Bob', np.nan, 'David', np.nan],  # Many nulls
            'email': ['alice@test.com', 'invalid', 'charlie@test.com', 'not_email', 'eve@test.com'],  # Invalid emails
            'age': [25, 200, -5, 40, 45],  # Invalid ages
            'salary': [50000, 60000, 70000, 80000, 90000]
        })
    
    def test_assess_completeness_full_data(self):
        """ทดสอบการประเมินความครบถ้วนของข้อมูลที่สมบูรณ์"""
        complete_data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E'],
            'col3': [10.1, 20.2, 30.3, 40.4, 50.5]
        })
        
        completeness = self.validator.assess_completeness(complete_data)
        
        self.assertIsInstance(completeness, dict)
        for col in complete_data.columns:
            self.assertEqual(completeness[col], 1.0)  # 100% complete
    
    def test_assess_completeness_missing_data(self):
        """ทดสอบการประเมินความครบถ้วนของข้อมูลที่มีข้อมูลหายไป"""
        data_with_nulls = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],  # 80% complete
            'col2': ['A', np.nan, np.nan, 'D', 'E'],  # 60% complete
            'col3': [10.1, 20.2, 30.3, 40.4, 50.5]  # 100% complete
        })
        
        completeness = self.validator.assess_completeness(data_with_nulls)
        
        self.assertEqual(completeness['col1'], 0.8)
        self.assertEqual(completeness['col2'], 0.6)
        self.assertEqual(completeness['col3'], 1.0)
    
    def test_assess_uniqueness_all_unique(self):
        """ทดสอบการประเมินความไม่ซ้ำของข้อมูลที่ไม่ซ้ำกัน"""
        unique_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E']
        })
        
        uniqueness = self.validator.assess_uniqueness(unique_data)
        
        for col in unique_data.columns:
            self.assertEqual(uniqueness[col], 1.0)  # 100% unique
    
    def test_assess_uniqueness_with_duplicates(self):
        """ทดสอบการประเมินความไม่ซ้ำของข้อมูลที่มีข้อมูลซ้ำ"""
        data_with_dups = pd.DataFrame({
            'id': [1, 1, 3, 3, 5],  # 60% unique (3 out of 5)
            'category': ['A', 'A', 'A', 'B', 'C']  # 60% unique (3 out of 5)
        })
        
        uniqueness = self.validator.assess_uniqueness(data_with_dups)
        
        self.assertEqual(uniqueness['id'], 0.6)
        self.assertEqual(uniqueness['category'], 0.6)
    
    def test_assess_consistency_email_valid(self):
        """ทดสอบการประเมินความสอดคล้องของอีเมลที่ถูกต้อง"""
        valid_emails = pd.DataFrame({
            'email': ['test@example.com', 'user@domain.org', 'admin@site.net']
        })
        
        consistency = self.validator.assess_consistency(valid_emails, 'email', 'email')
        
        self.assertEqual(consistency, 1.0)  # 100% consistent
    
    def test_assess_consistency_email_mixed(self):
        """ทดสอบการประเมินความสอดคล้องของอีเมลที่มีทั้งถูกและผิด"""
        mixed_emails = pd.DataFrame({
            'email': ['valid@test.com', 'invalid_email', 'another@valid.com', 'also_invalid', 'good@email.org']
        })
        
        consistency = self.validator.assess_consistency(mixed_emails, 'email', 'email')
        
        self.assertEqual(consistency, 0.6)  # 3 out of 5 valid = 60%
    
    def test_assess_consistency_phone_thai(self):
        """ทดสอบการประเมินความสอดคล้องของเบอร์โทรไทย"""
        thai_phones = pd.DataFrame({
            'phone': ['0812345678', '0823456789', 'invalid_phone', '+66812345678', '0634567890']
        })
        
        consistency = self.validator.assess_consistency(thai_phones, 'phone', 'phone_th')
        
        # Should have some valid Thai phone numbers
        self.assertGreater(consistency, 0)
        self.assertLessEqual(consistency, 1)
    
    def test_assess_accuracy_numeric_range(self):
        """ทดสอบการประเมินความถูกต้องของข้อมูลตัวเลขในช่วงที่กำหนด"""
        age_data = pd.DataFrame({
            'age': [25, 30, 150, 40, -5, 60, 70, 200, 35, 45]  # Some invalid ages
        })
        
        accuracy = self.validator.assess_accuracy(age_data, 'age', min_value=0, max_value=120)
        
        # Valid ages: 25, 30, 40, 60, 70, 35, 45 = 7 out of 10 = 70%
        self.assertEqual(accuracy, 0.7)
    
    def test_assess_accuracy_all_valid_range(self):
        """ทดสอบการประเมินความถูกต้องของข้อมูลที่อยู่ในช่วงที่ถูกต้องทั้งหมด"""
        valid_scores = pd.DataFrame({
            'score': [85, 92, 78, 95, 88, 76, 91, 83, 87, 94]
        })
        
        accuracy = self.validator.assess_accuracy(valid_scores, 'score', min_value=0, max_value=100)
        
        self.assertEqual(accuracy, 1.0)  # All scores are valid
    
    def test_validate_business_rules_age_salary(self):
        """ทดสอบการตรวจสอบกฎทางธุรกิจ: อายุและเงินเดือน"""
        def age_salary_rule(row):
            # Rule: Salary should increase with age (very simplified)
            return row['age'] >= 18 and row['salary'] > 0
        
        test_data = pd.DataFrame({
            'age': [25, 17, 35, 40],  # One underage
            'salary': [50000, 30000, 70000, -1000]  # One negative salary
        })
        
        validation_result = self.validator.validate_business_rules(test_data, [age_salary_rule])
        
        self.assertIsInstance(validation_result, dict)
        self.assertIn('passed', validation_result)
        self.assertIn('failed', validation_result)
        self.assertIn('pass_rate', validation_result)
        
        # Should have some failures
        self.assertGreater(len(validation_result['failed']), 0)
        self.assertLess(validation_result['pass_rate'], 1.0)
    
    def test_validate_business_rules_email_format(self):
        """ทดสอบการตรวจสอบกฎทางธุรกิจ: รูปแบบอีเมล"""
        def email_rule(row):
            import re
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            return bool(re.match(email_pattern, str(row['email'])))
        
        test_data = pd.DataFrame({
            'email': ['valid@test.com', 'invalid_email', 'another@valid.org', 'also_invalid']
        })
        
        validation_result = self.validator.validate_business_rules(test_data, [email_rule])
        
        self.assertEqual(validation_result['pass_rate'], 0.5)  # 2 out of 4 valid
    
    def test_calculate_quality_score_high_quality(self):
        """ทดสอบการคำนวณคะแนนคุณภาพสำหรับข้อมูลคุณภาพสูง"""
        quality_metrics = {
            'completeness': 0.95,
            'uniqueness': 0.98,
            'consistency': 0.97,
            'accuracy': 0.94,
            'validity': 0.96
        }
        
        score = self.validator.calculate_quality_score(quality_metrics)
        
        self.assertGreater(score, 0.9)  # Should be high quality
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_quality_score_low_quality(self):
        """ทดสอบการคำนวณคะแนนคุณภาพสำหรับข้อมูลคุณภาพต่ำ"""
        quality_metrics = {
            'completeness': 0.5,
            'uniqueness': 0.6,
            'consistency': 0.4,
            'accuracy': 0.3,
            'validity': 0.5
        }
        
        score = self.validator.calculate_quality_score(quality_metrics)
        
        self.assertLess(score, 0.6)  # Should be low quality
        self.assertGreaterEqual(score, 0.0)
    
    def test_validate_data_full_assessment(self):
        """ทดสอบการตรวจสอบข้อมูลแบบครบวงจร"""
        result = self.validator.validate_data(self.test_data)
        
        self.assertIsInstance(result, dict)
        self.assertIn('quality_score', result)
        self.assertIn('metrics', result)
        self.assertIn('issues', result)
        self.assertIn('recommendations', result)
        
        # Check metrics structure
        metrics = result['metrics']
        expected_metrics = ['completeness', 'uniqueness', 'consistency', 'accuracy', 'validity']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Quality score should be between 0 and 1
        self.assertGreaterEqual(result['quality_score'], 0)
        self.assertLessEqual(result['quality_score'], 1)
    
    def test_validate_data_poor_quality(self):
        """ทดสอบการตรวจสอบข้อมูลคุณภาพต่ำ"""
        result = self.validator.validate_data(self.poor_quality_data)
        
        # Should detect quality issues
        self.assertLess(result['quality_score'], 0.8)  # Poor quality
        self.assertGreater(len(result['issues']), 0)  # Should have issues
        self.assertGreater(len(result['recommendations']), 0)  # Should have recommendations
    
    def test_get_validation_report(self):
        """ทดสอบการสร้างรายงานการตรวจสอบ"""
        validation_result = self.validator.validate_data(self.test_data)
        report = self.validator.get_validation_report(validation_result)
        
        self.assertIsInstance(report, str)
        self.assertIn('Data Quality Report', report)
        self.assertIn('Quality Score', report)
        self.assertIn('Metrics', report)


if __name__ == '__main__':
    unittest.main()

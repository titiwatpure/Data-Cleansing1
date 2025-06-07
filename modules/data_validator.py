"""
โมดูลตรวจสอบข้อมูล (Data Validator Module)
=========================================

โมดูลนี้จัดการการตรวจสอบคุณภาพและความถูกต้องของข้อมูล
รวมถึงการสร้างรายงานการตรวจสอบ

ผู้พัฒนา: ทีมทำความสะอาดข้อมูล
วันที่: มิถุนายน 2568
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re


class DataValidator:
    """
    คลาสสำหรับการตรวจสอบคุณภาพข้อมูล
    
    คลาสนี้จัดการการตรวจสอบข้อมูลต่างๆ ประกอบด้วย:
    - การตรวจสอบความสมบูรณ์
    - การตรวจสอบความถูกต้อง
    - การตรวจสอบความสอดคล้อง
    - การตรวจสอบความไม่ซ้ำ
    - การประเมินคุณภาพข้อมูล
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        เริ่มต้นคลาส DataValidator
        
        Args:
            config (Dict): การตั้งค่าสำหรับการตรวจสอบ
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.quality_metrics = {}
        
    def validate_data(self, original_data: pd.DataFrame, 
                     cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        ตรวจสอบคุณภาพข้อมูลโดยเปรียบเทียบก่อนและหลังการทำความสะอาด
        
        Args:
            original_data (pd.DataFrame): ข้อมูลต้นฉบับ
            cleaned_data (pd.DataFrame): ข้อมูลที่ทำความสะอาดแล้ว
            
        Returns:
            Dict: ผลการตรวจสอบ
        """
        self.logger.info("🔍 เริ่มต้นการตรวจสอบคุณภาพข้อมูล")
        
        # ตรวจสอบคุณภาพข้อมูลเดี่ยว
        original_quality = self._assess_data_quality(original_data, "ข้อมูลต้นฉบับ")
        cleaned_quality = self._assess_data_quality(cleaned_data, "ข้อมูลที่ทำความสะอาดแล้ว")
        
        # เปรียบเทียบการเปลี่ยนแปลง
        comparison = self._compare_datasets(original_data, cleaned_data)
        
        # ตรวจสอบเฉพาะ
        specific_validations = self._perform_specific_validations(cleaned_data)
        
        # สร้างผลสรุป
        validation_summary = self._create_validation_summary(
            original_quality, cleaned_quality, comparison, specific_validations
        )
        
        self.validation_results = {
            'original_quality': original_quality,
            'cleaned_quality': cleaned_quality,
            'comparison': comparison,
            'specific_validations': specific_validations,
            'summary': validation_summary
        }
        
        self.logger.info("✅ การตรวจสอบคุณภาพข้อมูลเสร็จสิ้น")
        return self.validation_results
    
    def _assess_data_quality(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """ประเมินคุณภาพข้อมูล"""
        self.logger.info(f"📊 ประเมินคุณภาพ: {dataset_name}")
        
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        
        quality_assessment = {
            'dataset_name': dataset_name,
            'basic_info': {
                'rows': data.shape[0],
                'columns': data.shape[1],
                'total_cells': total_cells,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'completeness': {
                'missing_cells': missing_cells,
                'missing_percentage': (missing_cells / total_cells) * 100 if total_cells > 0 else 0,
                'complete_rows': data.dropna().shape[0],
                'complete_rows_percentage': (data.dropna().shape[0] / data.shape[0]) * 100 if data.shape[0] > 0 else 0
            },
            'uniqueness': self._assess_uniqueness(data),
            'consistency': self._assess_consistency(data),
            'accuracy': self._assess_accuracy(data),
            'validity': self._assess_validity(data)
        }
        
        # คำนวณคะแนนรวม
        quality_assessment['overall_score'] = self._calculate_quality_score(quality_assessment)
        
        return quality_assessment
    
    def _assess_uniqueness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ประเมินความไม่ซ้ำของข้อมูล"""
        duplicate_rows = data.duplicated().sum()
        
        uniqueness = {
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': (duplicate_rows / len(data)) * 100 if len(data) > 0 else 0,
            'unique_rows': len(data) - duplicate_rows,
            'uniqueness_score': ((len(data) - duplicate_rows) / len(data)) * 100 if len(data) > 0 else 100
        }
        
        # ตรวจสอบความซ้ำในแต่ละคอลัมน์
        column_uniqueness = {}
        for column in data.columns:
            unique_values = data[column].nunique()
            total_values = data[column].count()
            column_uniqueness[column] = {
                'unique_values': unique_values,
                'total_values': total_values,
                'uniqueness_ratio': unique_values / total_values if total_values > 0 else 0
            }
        
        uniqueness['column_uniqueness'] = column_uniqueness
        return uniqueness
    
    def _assess_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ประเมินความสอดคล้องของข้อมูล"""
        consistency_issues = []
        
        # ตรวจสอบรูปแบบข้อมูล
        for column in data.columns:
            if data[column].dtype == 'object':
                # ตรวจสอบรูปแบบในคอลัมน์ข้อความ
                pattern_consistency = self._check_pattern_consistency(data[column], column)
                if pattern_consistency['issues']:
                    consistency_issues.extend(pattern_consistency['issues'])
        
        # ตรวจสอบช่วงข้อมูลที่สมเหตุสมผล
        for column in data.select_dtypes(include=[np.number]).columns:
            range_check = self._check_reasonable_ranges(data[column], column)
            if range_check['issues']:
                consistency_issues.extend(range_check['issues'])
        
        # ตรวจสอบข้อมูลวันที่
        for column in data.select_dtypes(include=['datetime64[ns]']).columns:
            date_check = self._check_date_consistency(data[column], column)
            if date_check['issues']:
                consistency_issues.extend(date_check['issues'])
        
        consistency_score = max(0, 100 - len(consistency_issues) * 10)  # ลดคะแนน 10% ต่อปัญหา
        
        return {
            'issues': consistency_issues,
            'issue_count': len(consistency_issues),
            'consistency_score': consistency_score
        }
    
    def _assess_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ประเมินความถูกต้องของข้อมูล"""
        accuracy_issues = []
        
        # ตรวจสอบรูปแบบอีเมล
        email_columns = [col for col in data.columns if 'email' in col.lower() or 'อีเมล' in col.lower()]
        for column in email_columns:
            if column in data.columns:
                email_check = self._validate_email_format(data[column], column)
                if email_check['invalid_count'] > 0:
                    accuracy_issues.append(email_check)
        
        # ตรวจสอบรูปแบบหมายเลขโทรศัพท์
        phone_columns = [col for col in data.columns if any(keyword in col.lower() 
                        for keyword in ['phone', 'tel', 'mobile', 'โทรศัพท์', 'เบอร์'])]
        for column in phone_columns:
            if column in data.columns:
                phone_check = self._validate_phone_format(data[column], column)
                if phone_check['invalid_count'] > 0:
                    accuracy_issues.append(phone_check)
        
        # คำนวณคะแนนความถูกต้อง
        accuracy_score = max(0, 100 - len(accuracy_issues) * 15)  # ลดคะแนน 15% ต่อปัญหา
        
        return {
            'issues': accuracy_issues,
            'issue_count': len(accuracy_issues),
            'accuracy_score': accuracy_score
        }
    
    def _assess_validity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ประเมินความถูกต้องของข้อมูลตามกฎเกณฑ์"""
        validity_issues = []
        
        # ตรวจสอบค่าที่ออกนอกขอบเขต
        for column in data.select_dtypes(include=[np.number]).columns:
            # ตรวจสอบค่าลบในคอลัมน์ที่ไม่ควรเป็นลบ
            if any(keyword in column.lower() for keyword in ['age', 'price', 'amount', 'count', 'quantity']):
                negative_count = (data[column] < 0).sum()
                if negative_count > 0:
                    validity_issues.append({
                        'column': column,
                        'issue': f'พบค่าลบ {negative_count} จุด ในคอลัมน์ที่ไม่ควรเป็นลบ',
                        'severity': 'สูง'
                    })
            
            # ตรวจสอบค่าที่สูงผิดปกติ (outliers)
            if len(data[column].dropna()) > 0:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = len(data[(data[column] < Q1 - 3 * IQR) | (data[column] > Q3 + 3 * IQR)])
                
                if outlier_count > len(data) * 0.05:  # มากกว่า 5%
                    validity_issues.append({
                        'column': column,
                        'issue': f'พบข้อมูลผิดปกติมาก {outlier_count} จุด ({outlier_count/len(data)*100:.1f}%)',
                        'severity': 'กลาง'
                    })
        
        # คำนวณคะแนนความถูกต้องตามกฎ
        validity_score = max(0, 100 - len(validity_issues) * 12)  # ลดคะแนน 12% ต่อปัญหา
        
        return {
            'issues': validity_issues,
            'issue_count': len(validity_issues),
            'validity_score': validity_score
        }
    
    def _check_pattern_consistency(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """ตรวจสอบความสอดคล้องของรูปแบบ"""
        issues = []
        
        # ตรวจสอบรูปแบบการใช้ตัวพิมพ์
        if series.dtype == 'object':
            mixed_case_count = sum(1 for x in series.dropna() if isinstance(x, str) and 
                                 x != x.upper() and x != x.lower() and any(c.isalpha() for c in x))
            
            if mixed_case_count > len(series) * 0.1:  # มากกว่า 10%
                issues.append({
                    'column': column_name,
                    'issue': f'รูปแบบตัวพิมพ์ไม่สอดคล้อง {mixed_case_count} จุด',
                    'severity': 'ต่ำ'
                })
        
        return {'issues': issues}
    
    def _check_reasonable_ranges(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """ตรวจสอบช่วงข้อมูลที่สมเหตุสมผล"""
        issues = []
        
        # ตรวจสอบอายุ
        if 'age' in column_name.lower() or 'อายุ' in column_name.lower():
            unreasonable_age = ((series < 0) | (series > 150)).sum()
            if unreasonable_age > 0:
                issues.append({
                    'column': column_name,
                    'issue': f'อายุที่ไม่สมเหตุสมผล {unreasonable_age} จุด',
                    'severity': 'สูง'
                })
        
        # ตรวจสอบเปอร์เซ็นต์
        if 'percent' in column_name.lower() or 'เปอร์เซ็นต์' in column_name.lower():
            invalid_percent = ((series < 0) | (series > 100)).sum()
            if invalid_percent > 0:
                issues.append({
                    'column': column_name,
                    'issue': f'เปอร์เซ็นต์ที่ไม่ถูกต้อง {invalid_percent} จุด',
                    'severity': 'สูง'
                })
        
        return {'issues': issues}
    
    def _check_date_consistency(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """ตรวจสอบความสอดคล้องของวันที่"""
        issues = []
        
        # ตรวจสอบวันที่ในอนาคต
        future_dates = (series > datetime.now()).sum()
        if future_dates > 0:
            issues.append({
                'column': column_name,
                'issue': f'วันที่ในอนาคต {future_dates} จุด',
                'severity': 'กลาง'
            })
        
        # ตรวจสอบวันที่ที่เก่าเกินไป
        very_old_dates = (series < datetime(1900, 1, 1)).sum()
        if very_old_dates > 0:
            issues.append({
                'column': column_name,
                'issue': f'วันที่เก่าเกินไป {very_old_dates} จุด',
                'severity': 'กลาง'
            })
        
        return {'issues': issues}
    
    def _validate_email_format(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """ตรวจสอบรูปแบบอีเมล"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        valid_emails = series.astype(str).str.match(email_pattern).sum()
        total_emails = series.count()
        invalid_count = total_emails - valid_emails
        
        return {
            'column': column_name,
            'issue': f'อีเมลรูปแบบไม่ถูกต้อง {invalid_count} จุด จากทั้งหมด {total_emails} จุด',
            'invalid_count': invalid_count,
            'validity_percentage': (valid_emails / total_emails * 100) if total_emails > 0 else 0,
            'severity': 'กลาง' if invalid_count > total_emails * 0.1 else 'ต่ำ'
        }
    
    def _validate_phone_format(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """ตรวจสอบรูปแบบหมายเลขโทรศัพท์"""
        phone_patterns = [
            r'^\d{3}-\d{3}-\d{4}$',      # 123-456-7890
            r'^\(\d{3}\)\s\d{3}-\d{4}$', # (123) 456-7890
            r'^\d{10}$',                 # 1234567890
            r'^0\d{8,9}$',               # 0812345678 (รูปแบบไทย)
        ]
        
        valid_phones = 0
        for phone in series.dropna():
            phone_str = str(phone).strip()
            if any(re.match(pattern, phone_str) for pattern in phone_patterns):
                valid_phones += 1
        
        total_phones = series.count()
        invalid_count = total_phones - valid_phones
        
        return {
            'column': column_name,
            'issue': f'หมายเลขโทรศัพท์รูปแบบไม่ถูกต้อง {invalid_count} จุด จากทั้งหมด {total_phones} จุด',
            'invalid_count': invalid_count,
            'validity_percentage': (valid_phones / total_phones * 100) if total_phones > 0 else 0,
            'severity': 'กลาง' if invalid_count > total_phones * 0.1 else 'ต่ำ'
        }
    
    def _compare_datasets(self, original: pd.DataFrame, cleaned: pd.DataFrame) -> Dict[str, Any]:
        """เปรียบเทียบข้อมูลก่อนและหลังการทำความสะอาด"""
        comparison = {
            'rows_changed': {
                'original': len(original),
                'cleaned': len(cleaned),
                'difference': len(cleaned) - len(original),
                'percentage_change': ((len(cleaned) - len(original)) / len(original) * 100) if len(original) > 0 else 0
            },
            'columns_changed': {
                'original': len(original.columns),
                'cleaned': len(cleaned.columns),
                'difference': len(cleaned.columns) - len(original.columns),
                'new_columns': list(set(cleaned.columns) - set(original.columns)),
                'removed_columns': list(set(original.columns) - set(cleaned.columns))
            },
            'missing_data_improvement': {
                'original_missing': original.isnull().sum().sum(),
                'cleaned_missing': cleaned.isnull().sum().sum(),
                'improvement': original.isnull().sum().sum() - cleaned.isnull().sum().sum()
            }
        }
        
        return comparison
    
    def _perform_specific_validations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบเฉพาะเจาะจง"""
        validations = {}
        
        # ตรวจสอบการกระจายของข้อมูล
        validations['data_distribution'] = self._check_data_distribution(data)
        
        # ตรวจสอบความสัมพันธ์ระหว่างคอลัมน์
        validations['column_relationships'] = self._check_column_relationships(data)
        
        # ตรวจสอบข้อมูลที่มีความหมาย
        validations['business_rules'] = self._check_business_rules(data)
        
        return validations
    
    def _check_data_distribution(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบการกระจายของข้อมูล"""
        distribution_info = {}
        
        for column in data.select_dtypes(include=[np.number]).columns:
            series = data[column].dropna()
            if len(series) > 0:
                distribution_info[column] = {
                    'mean': float(series.mean()),
                    'median': float(series.median()),
                    'std': float(series.std()),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis())
                }
        
        return distribution_info
    
    def _check_column_relationships(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบความสัมพันธ์ระหว่างคอลัมน์"""
        relationships = {}
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            correlation_matrix = data[numeric_columns].corr()
            
            # หาความสัมพันธ์ที่แรง (> 0.8 หรือ < -0.8)
            strong_correlations = []
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i < j:  # หลีกเลี่ยงการซ้ำ
                        corr_value = correlation_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.8:
                            strong_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': float(corr_value)
                            })
            
            relationships['strong_correlations'] = strong_correlations
        
        return relationships
    
    def _check_business_rules(self, data: pd.DataFrame) -> Dict[str, Any]:
        """ตรวจสอบกฎทางธุรกิจ"""
        violations = []
        
        # ตัวอย่างกฎทางธุรกิจ
        # กฎ 1: อายุต้องมากกว่าวันเกิด
        age_columns = [col for col in data.columns if 'age' in col.lower() or 'อายุ' in col.lower()]
        birth_columns = [col for col in data.columns if any(keyword in col.lower() 
                        for keyword in ['birth', 'born', 'เกิด'])]
        
        if age_columns and birth_columns:
            for age_col in age_columns:
                for birth_col in birth_columns:
                    if age_col in data.columns and birth_col in data.columns:
                        # ตรวจสอบความสอดคล้องระหว่างอายุและวันเกิด
                        # (การตรวจสอบจริงจะซับซ้อนกว่านี้)
                        violations.append({
                            'rule': f'ความสอดคล้องระหว่าง {age_col} และ {birth_col}',
                            'status': 'ต้องตรวจสอบเพิ่มเติม'
                        })
        
        return {'violations': violations}
    
    def _calculate_quality_score(self, quality_assessment: Dict[str, Any]) -> float:
        """คำนวณคะแนนคุณภาพรวม"""
        # น้ำหัก: ความสมบูรณ์ 30%, ความไม่ซ้ำ 25%, ความสอดคล้อง 25%, ความถูกต้อง 20%
        completeness_score = 100 - quality_assessment['completeness']['missing_percentage']
        uniqueness_score = quality_assessment['uniqueness']['uniqueness_score']
        consistency_score = quality_assessment['consistency']['consistency_score']
        accuracy_score = quality_assessment['accuracy']['accuracy_score']
        
        overall_score = (
            completeness_score * 0.30 +
            uniqueness_score * 0.25 +
            consistency_score * 0.25 +
            accuracy_score * 0.20
        )
        
        return round(overall_score, 1)
    
    def _create_validation_summary(self, original_quality: Dict, cleaned_quality: Dict,
                                 comparison: Dict, specific_validations: Dict) -> Dict[str, Any]:
        """สร้างสรุปการตรวจสอบ"""
        return {
            'quality_improvement': {
                'original_score': original_quality['overall_score'],
                'cleaned_score': cleaned_quality['overall_score'],
                'improvement': cleaned_quality['overall_score'] - original_quality['overall_score']
            },
            'data_changes': {
                'rows_removed': max(0, -comparison['rows_changed']['difference']),
                'rows_added': max(0, comparison['rows_changed']['difference']),
                'columns_added': len(comparison['columns_changed']['new_columns']),
                'missing_data_reduced': comparison['missing_data_improvement']['improvement']
            },
            'recommendations': self._generate_recommendations(cleaned_quality, specific_validations),
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _generate_recommendations(self, quality_assessment: Dict, 
                                specific_validations: Dict) -> List[str]:
        """สร้างคำแนะนำสำหรับการปรับปรุงข้อมูล"""
        recommendations = []
        
        # คำแนะนำตามคะแนนคุณภาพ
        if quality_assessment['overall_score'] < 80:
            recommendations.append("คุณภาพข้อมูลยังต่ำกว่า 80% ควรปรับปรุงเพิ่มเติม")
        
        # คำแนะนำตามความสมบูรณ์
        if quality_assessment['completeness']['missing_percentage'] > 10:
            recommendations.append("ข้อมูลที่ขาดหายไปมากกว่า 10% ควรพิจารณาการเติมข้อมูลเพิ่มเติม")
        
        # คำแนะนำตามความซ้ำ
        if quality_assessment['uniqueness']['duplicate_percentage'] > 5:
            recommendations.append("พบข้อมูลซ้ำมากกว่า 5% ควรตรวจสอบการลบข้อมูลซ้ำ")
        
        # คำแนะนำตามปัญหาความสอดคล้อง
        if quality_assessment['consistency']['issue_count'] > 0:
            recommendations.append(f"พบปัญหาความสอดคล้อง {quality_assessment['consistency']['issue_count']} จุด ควรแก้ไข")
        
        # คำแนะนำตามปัญหาความถูกต้อง
        if quality_assessment['accuracy']['issue_count'] > 0:
            recommendations.append(f"พบปัญหาความถูกต้อง {quality_assessment['accuracy']['issue_count']} จุด ควรตรวจสอบ")
        
        return recommendations
    
    def get_validation_results(self) -> Dict[str, Any]:
        """ส่งคืนผลการตรวจสอบ"""
        return self.validation_results
    
    def export_validation_report(self, file_path: str, format: str = 'json'):
        """ส่งออกรายงานการตรวจสอบ"""
        if format.lower() == 'json':
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"ส่งออกรายงานการตรวจสอบไปยัง: {file_path}")

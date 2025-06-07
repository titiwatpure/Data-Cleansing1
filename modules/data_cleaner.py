"""
โมดูลทำความสะอาดข้อมูล (Data Cleaner Module)
=============================================

โมดูลนี้ประกอบด้วยคลาสและฟังก์ชันสำหรับการทำความสะอาดข้อมูล
รองรับการดำเนินการทำความสะอาดต่างๆ ตามมาตรฐาน PEP 8

ผู้พัฒนา: ทีมทำความสะอาดข้อมูล
วันที่: มิถุนายน 2568
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union
import re
from datetime import datetime


class DataCleaner:
    """
    คลาสหลักสำหรับการทำความสะอาดข้อมูล
    
    คลาสนี้จัดการกระบวนการทำความสะอาดข้อมูลทั้งหมด ประกอบด้วย:
    - การลบข้อมูลที่ขาดหายไป
    - การเติมค่าข้อมูลที่ขาด
    - การลบข้อมูลซ้ำ
    - การแก้ไขรูปแบบข้อมูล
    - การตรวจสอบข้อมูลผิดปกติ
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        เริ่มต้นคลาส DataCleaner
        
        Args:
            config (Dict): การตั้งค่าสำหรับการทำความสะอาด
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.cleaning_log = []  # บันทึกขั้นตอนการทำความสะอาด
        
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        เตรียมข้อมูลเบื้องต้นก่อนการทำความสะอาด
        
        Args:
            data (pd.DataFrame): ข้อมูลต้นฉบับ
            
        Returns:
            pd.DataFrame: ข้อมูลที่เตรียมแล้ว
        """
        self.logger.info("🔧 เริ่มต้นการเตรียมข้อมูลเบื้องต้น")
        
        # สำเนาข้อมูลเพื่อป้องกันการแก้ไขต้นฉบับ
        processed_data = data.copy()
        
        # แสดงข้อมูลเบื้องต้น
        self._log_data_info("ข้อมูลก่อนการเตรียม", processed_data)
        
        # กำหนดประเภทข้อมูลเบื้องต้น
        processed_data = self._detect_and_convert_types(processed_data)
        
        # ทำความสะอาดชื่อคอลัมน์
        processed_data = self._clean_column_names(processed_data)
        
        self.logger.info("✅ การเตรียมข้อมูลเบื้องต้นเสร็จสิ้น")
        return processed_data
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ดำเนินการทำความสะอาดข้อมูลหลัก
        
        Args:
            data (pd.DataFrame): ข้อมูลที่เตรียมแล้ว
            
        Returns:
            pd.DataFrame: ข้อมูลที่ทำความสะอาดแล้ว
        """
        self.logger.info("🧹 เริ่มต้นการทำความสะอาดข้อมูล")
        
        cleaned_data = data.copy()
        
        # ขั้นตอนการทำความสะอาด
        cleaning_steps = [
            ("ลบแถวว่างเปล่า", self._remove_empty_rows),
            ("จัดการข้อมูลที่ขาด", self._handle_missing_data),
            ("ลบข้อมูลซ้ำ", self._remove_duplicates),
            ("แก้ไขรูปแบบข้อมูล", self._standardize_formats),
            ("ตรวจสอบข้อมูลผิดปกติ", self._detect_outliers),
            ("ตรวจสอบความสอดคล้อง", self._validate_consistency)
        ]
        
        for step_name, step_function in cleaning_steps:
            self.logger.info(f"📝 {step_name}...")
            try:
                cleaned_data = step_function(cleaned_data)
                self.cleaning_log.append(f"✅ {step_name} - เสร็จสิ้น")
            except Exception as e:
                error_msg = f"❌ {step_name} - ข้อผิดพลาด: {str(e)}"
                self.logger.error(error_msg)
                self.cleaning_log.append(error_msg)
                
        self._log_data_info("ข้อมูลหลังการทำความสะอาด", cleaned_data)
        self.logger.info("✅ การทำความสะอาดข้อมูลเสร็จสิ้น")
        
        return cleaned_data
    
    def _detect_and_convert_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบและแปลงประเภทข้อมูลอัตโนมัติ"""
        self.logger.info("🔍 ตรวจสอบและแปลงประเภทข้อมูล")
        
        for column in data.columns:
            # ตรวจสอบข้อมูลวันที่
            if self._is_date_column(data[column]):
                data[column] = pd.to_datetime(data[column], errors='coerce')
                self.logger.info(f"📅 แปลงคอลัมน์ '{column}' เป็นประเภทวันที่")
                
            # ตรวจสอบข้อมูลตัวเลข
            elif self._is_numeric_column(data[column]):
                data[column] = pd.to_numeric(data[column], errors='coerce')
                self.logger.info(f"🔢 แปลงคอลัมน์ '{column}' เป็นประเภทตัวเลข")
                
        return data
    
    def _clean_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """ทำความสะอาดชื่อคอลัมน์"""
        self.logger.info("📝 ทำความสะอาดชื่อคอลัมน์")
        
        # ลบช่องว่างข้างหน้าและข้างหลัง
        data.columns = data.columns.str.strip()
        
        # แทนที่ช่องว่างด้วย underscore
        data.columns = data.columns.str.replace(' ', '_', regex=False)
        
        # ลบอักขระพิเศษ
        data.columns = data.columns.str.replace(r'[^\w]', '_', regex=True)
        
        # แปลงเป็นตัวพิมพ์เล็ก
        data.columns = data.columns.str.lower()
        
        return data
    
    def _remove_empty_rows(self, data: pd.DataFrame) -> pd.DataFrame:
        """ลบแถวที่ว่างเปล่าทั้งหมด"""
        initial_rows = len(data)
        data_cleaned = data.dropna(how='all')
        removed_rows = initial_rows - len(data_cleaned)
        
        if removed_rows > 0:
            self.logger.info(f"🗑️ ลบแถวว่างเปล่า: {removed_rows} แถว")
            
        return data_cleaned
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """จัดการข้อมูลที่ขาดหายไป"""
        missing_summary = data.isnull().sum()
        
        for column in data.columns:
            missing_count = missing_summary[column]
            if missing_count > 0:
                missing_percent = (missing_count / len(data)) * 100
                self.logger.info(f"📊 คอลัมน์ '{column}': ข้อมูลขาด {missing_count} จุด ({missing_percent:.1f}%)")
                
                # เติมค่าตามประเภทข้อมูล
                if data[column].dtype in ['int64', 'float64']:
                    # เติมด้วยค่าเฉลี่ยสำหรับตัวเลข
                    fill_value = data[column].mean()
                    data[column].fillna(fill_value, inplace=True)
                    self.logger.info(f"🔧 เติมค่าเฉลี่ย ({fill_value:.2f}) ในคอลัมน์ '{column}'")
                    
                elif data[column].dtype == 'object':
                    # เติมด้วยค่าที่พบบ่อยที่สุดสำหรับข้อความ
                    fill_value = data[column].mode().iloc[0] if not data[column].mode().empty else 'Unknown'
                    data[column].fillna(fill_value, inplace=True)
                    self.logger.info(f"📝 เติมค่า '{fill_value}' ในคอลัมน์ '{column}'")
                    
                elif data[column].dtype == 'datetime64[ns]':
                    # เติมด้วยค่ากลางสำหรับวันที่
                    fill_value = data[column].median()
                    data[column].fillna(fill_value, inplace=True)
                    self.logger.info(f"📅 เติมค่ากลาง ({fill_value}) ในคอลัมน์ '{column}'")
                    
        return data
    
    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """ลบข้อมูลซ้ำ"""
        initial_rows = len(data)
        data_deduplicated = data.drop_duplicates()
        duplicate_count = initial_rows - len(data_deduplicated)
        
        if duplicate_count > 0:
            self.logger.info(f"🔄 ลบข้อมูลซ้ำ: {duplicate_count} แถว")
        else:
            self.logger.info("✅ ไม่พบข้อมูลซ้ำ")
            
        return data_deduplicated
    
    def _standardize_formats(self, data: pd.DataFrame) -> pd.DataFrame:
        """แก้ไขรูปแบบข้อมูลให้มาตรฐาน"""
        for column in data.columns:
            if data[column].dtype == 'object':
                # ลบช่องว่างข้างหน้าและข้างหลัง
                data[column] = data[column].astype(str).str.strip()
                
                # ตรวจสอบว่าเป็นข้อมูลอีเมลหรือไม่
                if self._is_email_column(data[column]):
                    data[column] = self._standardize_emails(data[column])
                    self.logger.info(f"📧 แก้ไขรูปแบบอีเมลในคอลัมน์ '{column}'")
                    
                # ตรวจสอบว่าเป็นข้อมูลหมายเลขโทรศัพท์หรือไม่
                elif self._is_phone_column(data[column]):
                    data[column] = self._standardize_phones(data[column])
                    self.logger.info(f"📞 แก้ไขรูปแบบหมายเลขโทรศัพท์ในคอลัมน์ '{column}'")
                    
        return data
    
    def _detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบข้อมูลผิดปกติ (Outliers)"""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            # ใช้วิธี IQR (Interquartile Range)
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            
            if len(outliers) > 0:
                outlier_count = len(outliers)
                self.logger.info(f"🎯 พบข้อมูลผิดปกติในคอลัมน์ '{column}': {outlier_count} จุด")
                
                # บันทึกข้อมูลผิดปกติ (ไม่ลบออก แค่แจ้งเตือน)
                self.cleaning_log.append(f"⚠️ คอลัมน์ '{column}': {outlier_count} ข้อมูลผิดปกติ")
                
        return data
    
    def _validate_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """ตรวจสอบความสอดคล้องของข้อมูล"""
        # ตรวจสอบความสอดคล้องของข้อมูลประเภทต่างๆ
        consistency_issues = []
        
        # ตรวจสอบรูปแบบวันที่
        date_columns = data.select_dtypes(include=['datetime64[ns]']).columns
        for column in date_columns:
            future_dates = data[data[column] > datetime.now()]
            if len(future_dates) > 0:
                issue = f"คอลัมน์ '{column}': พบวันที่ในอนาคต {len(future_dates)} จุด"
                consistency_issues.append(issue)
                self.logger.warning(f"⚠️ {issue}")
        
        # ตรวจสอบค่าติดลบในคอลัมน์ที่ไม่ควรติดลบ
        positive_columns = [col for col in data.columns if any(keyword in col.lower() 
                          for keyword in ['age', 'price', 'amount', 'quantity', 'count'])]
        
        for column in positive_columns:
            if column in data.columns and data[column].dtype in ['int64', 'float64']:
                negative_values = data[data[column] < 0]
                if len(negative_values) > 0:
                    issue = f"คอลัมน์ '{column}': พบค่าติดลบ {len(negative_values)} จุด"
                    consistency_issues.append(issue)
                    self.logger.warning(f"⚠️ {issue}")
        
        if consistency_issues:
            self.cleaning_log.extend([f"⚠️ {issue}" for issue in consistency_issues])
        else:
            self.logger.info("✅ ข้อมูลมีความสอดคล้อง")
            
        return data
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """ตรวจสอบว่าคอลัมน์เป็นข้อมูลวันที่หรือไม่"""
        if series.dtype == 'object':
            sample = series.dropna().head(10)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # DD/MM/YYYY
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            ]
            
            for item in sample:
                if isinstance(item, str):
                    for pattern in date_patterns:
                        if re.match(pattern, item.strip()):
                            return True
        return False
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """ตรวจสอบว่าคอลัมน์เป็นข้อมูลตัวเลขหรือไม่"""
        if series.dtype == 'object':
            try:
                pd.to_numeric(series.dropna().head(10), errors='raise')
                return True
            except (ValueError, TypeError):
                return False
        return False
    
    def _is_email_column(self, series: pd.Series) -> bool:
        """ตรวจสอบว่าคอลัมน์เป็นข้อมูลอีเมลหรือไม่"""
        sample = series.dropna().head(10)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        for item in sample:
            if isinstance(item, str) and re.match(email_pattern, item.strip()):
                return True
        return False
    
    def _is_phone_column(self, series: pd.Series) -> bool:
        """ตรวจสอบว่าคอลัมน์เป็นข้อมูลหมายเลขโทรศัพท์หรือไม่"""
        sample = series.dropna().head(10)
        phone_patterns = [
            r'\d{3}-\d{3}-\d{4}',      # 123-456-7890
            r'\(\d{3}\)\s\d{3}-\d{4}', # (123) 456-7890
            r'\d{10}',                 # 1234567890
        ]
        
        for item in sample:
            if isinstance(item, str):
                for pattern in phone_patterns:
                    if re.match(pattern, item.strip()):
                        return True
        return False
    
    def _standardize_emails(self, series: pd.Series) -> pd.Series:
        """แก้ไขรูปแบบอีเมลให้มาตรฐาน"""
        return series.str.lower().str.strip()
    
    def _standardize_phones(self, series: pd.Series) -> pd.Series:
        """แก้ไขรูปแบบหมายเลขโทรศัพท์ให้มาตรฐาน"""
        # ลบอักขระที่ไม่ใช่ตัวเลข
        cleaned = series.str.replace(r'[^\d]', '', regex=True)
        
        # จัดรูปแบบเป็น XXX-XXX-XXXX
        formatted = cleaned.apply(lambda x: f"{x[:3]}-{x[3:6]}-{x[6:]}" if len(x) == 10 else x)
        
        return formatted
    
    def _log_data_info(self, stage: str, data: pd.DataFrame):
        """บันทึกข้อมูลสถิติของข้อมูล"""
        self.logger.info(f"📊 {stage}:")
        self.logger.info(f"   - จำนวนแถว: {len(data):,}")
        self.logger.info(f"   - จำนวนคอลัมน์: {len(data.columns)}")
        self.logger.info(f"   - ข้อมูลที่ขาด: {data.isnull().sum().sum():,} จุด")
        self.logger.info(f"   - ขนาดข้อมูล: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    
    def get_cleaning_summary(self) -> List[str]:
        """ส่งคืนสรุปการทำความสะอาด"""
        return self.cleaning_log.copy()
    
    def reset_log(self):
        """รีเซ็ตบันทึกการทำความสะอาด"""
        self.cleaning_log = []

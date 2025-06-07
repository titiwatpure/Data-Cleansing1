"""
โมดูลแปลงข้อมูล (Data Transformer Module)
========================================

โมดูลนี้จัดการการแปลงข้อมูลหลังจากการทำความสะอาด
รวมถึงการสร้างฟีเจอร์ใหม่และการแปลงรูปแบบข้อมูล

ผู้พัฒนา: ทีมทำความสะอาดข้อมูล
วันที่: มิถุนายน 2568
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import re


class DataTransformer:
    """
    คลาสสำหรับการแปลงข้อมูล
    
    คลาสนี้จัดการการแปลงข้อมูลต่างๆ ประกอบด้วย:
    - การสร้างคอลัมน์ใหม่
    - การรวมคอลัมน์
    - การแปลงรหัสหรือค่า
    - การแปลงประเภทข้อมูล
    - การสร้างฟีเจอร์ (Feature Engineering)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        เริ่มต้นคลาส DataTransformer
        
        Args:
            config (Dict): การตั้งค่าสำหรับการแปลงข้อมูล
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.transformation_log = []  # บันทึกขั้นตอนการแปลง
        self.value_mappings = {}  # เก็บการแมปค่าต่างๆ
        
    def transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        ดำเนินการแปลงข้อมูลหลัก
        
        Args:
            data (pd.DataFrame): ข้อมูลที่ทำความสะอาดแล้ว
            
        Returns:
            pd.DataFrame: ข้อมูลที่แปลงแล้ว
        """
        self.logger.info("🔄 เริ่มต้นการแปลงข้อมูล")
        
        transformed_data = data.copy()
        
        # ขั้นตอนการแปลงข้อมูล
        transformation_steps = [
            ("สร้างฟีเจอร์ใหม่", self._create_new_features),
            ("แปลงรหัสและค่า", self._map_values),
            ("รวมและแยกคอลัมน์", self._merge_split_columns),
            ("แปลงประเภทข้อมูลขั้นสุดท้าย", self._final_type_conversion),
            ("การปรับมาตรฐาน", self._normalize_standardize)
        ]
        
        for step_name, step_function in transformation_steps:
            self.logger.info(f"🔧 {step_name}...")
            try:
                transformed_data = step_function(transformed_data)
                self.transformation_log.append(f"✅ {step_name} - เสร็จสิ้น")
            except Exception as e:
                error_msg = f"❌ {step_name} - ข้อผิดพลาด: {str(e)}"
                self.logger.error(error_msg)
                self.transformation_log.append(error_msg)
                
        self._log_transformation_summary(data, transformed_data)
        self.logger.info("✅ การแปลงข้อมูลเสร็จสิ้น")
        
        return transformed_data
    
    def _create_new_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """สร้างฟีเจอร์ใหม่จากข้อมูลที่มีอยู่"""
        self.logger.info("🏗️ สร้างฟีเจอร์ใหม่")
        
        # สร้างฟีเจอร์วันที่
        date_columns = data.select_dtypes(include=['datetime64[ns]']).columns
        for column in date_columns:
            self._create_date_features(data, column)
            
        # สร้างฟีเจอร์ข้อความ
        text_columns = data.select_dtypes(include=['object']).columns
        for column in text_columns:
            self._create_text_features(data, column)
            
        # สร้างฟีเจอร์ตัวเลข
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            self._create_numeric_features(data, numeric_columns)
            
        return data
    
    def _create_date_features(self, data: pd.DataFrame, column: str):
        """สร้างฟีเจอร์จากคอลัมน์วันที่"""
        base_name = column.replace('_date', '').replace('_time', '')
        
        # แยกส่วนประกอบของวันที่
        data[f'{base_name}_year'] = data[column].dt.year
        data[f'{base_name}_month'] = data[column].dt.month
        data[f'{base_name}_day'] = data[column].dt.day
        data[f'{base_name}_weekday'] = data[column].dt.weekday
        data[f'{base_name}_quarter'] = data[column].dt.quarter
        
        # สร้างฟีเจอร์เพิ่มเติม
        data[f'{base_name}_is_weekend'] = data[column].dt.weekday >= 5
        data[f'{base_name}_days_from_today'] = (datetime.now() - data[column]).dt.days
        
        self.logger.info(f"📅 สร้างฟีเจอร์วันที่สำหรับคอลัมน์ '{column}'")
    
    def _create_text_features(self, data: pd.DataFrame, column: str):
        """สร้างฟีเจอร์จากคอลัมน์ข้อความ"""
        # ความยาวข้อความ
        data[f'{column}_length'] = data[column].astype(str).str.len()
        
        # จำนวนคำ
        data[f'{column}_word_count'] = data[column].astype(str).str.split().str.len()
        
        # มีตัวเลขหรือไม่
        data[f'{column}_has_numbers'] = data[column].astype(str).str.contains(r'\d', regex=True)
        
        # มีอักขระพิเศษหรือไม่
        data[f'{column}_has_special'] = data[column].astype(str).str.contains(r'[^a-zA-Z0-9\s]', regex=True)
        
        # เป็นตัวพิมพ์ใหญ่ทั้งหมดหรือไม่
        data[f'{column}_is_upper'] = data[column].astype(str).str.isupper()
        
        self.logger.info(f"📝 สร้างฟีเจอร์ข้อความสำหรับคอลัมน์ '{column}'")
    
    def _create_numeric_features(self, data: pd.DataFrame, numeric_columns: pd.Index):
        """สร้างฟีเจอร์จากคอลัมน์ตัวเลข"""
        # สร้างฟีเจอร์การรวม (Aggregation Features)
        if len(numeric_columns) >= 2:
            # ผลรวม
            data['total_sum'] = data[numeric_columns].sum(axis=1)
            
            # ค่าเฉลี่ย
            data['average'] = data[numeric_columns].mean(axis=1)
            
            # ค่าสูงสุดและต่ำสุด
            data['max_value'] = data[numeric_columns].max(axis=1)
            data['min_value'] = data[numeric_columns].min(axis=1)
            
            # ช่วงค่า (Range)
            data['value_range'] = data['max_value'] - data['min_value']
            
            self.logger.info("🔢 สร้างฟีเจอร์การรวมจากคอลัมน์ตัวเลข")
    
    def _map_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """แปลงรหัสและค่าต่างๆ"""
        self.logger.info("🗺️ แปลงรหัสและค่า")
        
        # แมปค่าทั่วไป
        common_mappings = {
            # แมปค่าบูลีน
            'boolean_mappings': {
                'yes': True, 'no': False, 'y': True, 'n': False,
                'true': True, 'false': False, '1': True, '0': False,
                'ใช่': True, 'ไม่ใช่': False, 'ใช้': True, 'ไม่ใช้': False
            },
            
            # แมประดับการศึกษา
            'education_mappings': {
                'ประถม': 1, 'มัธยม': 2, 'ปวช': 3, 'ปวส': 4,
                'ปริญญาตรี': 5, 'ปริญญาโท': 6, 'ปริญญาเอก': 7,
                'primary': 1, 'secondary': 2, 'bachelor': 5, 'master': 6, 'phd': 7
            },
            
            # แมปขนาดธุรกิจ
            'size_mappings': {
                'เล็ก': 1, 'กลาง': 2, 'ใหญ่': 3,
                'small': 1, 'medium': 2, 'large': 3,
                's': 1, 'm': 2, 'l': 3, 'xl': 4
            }
        }
        
        # ใช้การแมปกับคอลัมน์ที่เหมาะสม
        for column in data.columns:
            if data[column].dtype == 'object':
                column_lower = column.lower()
                
                # ตรวจสอบและใช้การแมปที่เหมาะสม
                if any(keyword in column_lower for keyword in ['bool', 'flag', 'is_', 'has_']):
                    data[column] = self._apply_mapping(data[column], common_mappings['boolean_mappings'])
                    self.logger.info(f"✅ แมปค่าบูลีนในคอลัมน์ '{column}'")
                    
                elif any(keyword in column_lower for keyword in ['education', 'degree', 'การศึกษา']):
                    data[column] = self._apply_mapping(data[column], common_mappings['education_mappings'])
                    self.logger.info(f"📚 แมประดับการศึกษาในคอลัมน์ '{column}'")
                    
                elif any(keyword in column_lower for keyword in ['size', 'ขนาด']):
                    data[column] = self._apply_mapping(data[column], common_mappings['size_mappings'])
                    self.logger.info(f"📏 แมปขนาดในคอลัมน์ '{column}'")
        
        return data
    
    def _apply_mapping(self, series: pd.Series, mapping: Dict) -> pd.Series:
        """ใช้การแมปกับ Series"""
        # แปลงเป็นตัวพิมพ์เล็กเพื่อการเปรียบเทียบ
        series_mapped = series.astype(str).str.lower().map(mapping)
        
        # คืนค่าเดิมถ้าไม่พบการแมป
        return series_mapped.fillna(series)
    
    def _merge_split_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """รวมและแยกคอลัมน์"""
        self.logger.info("🔗 รวมและแยกคอลัมน์")
        
        # รวมคอลัมน์ชื่อ (ถ้ามี)
        name_columns = [col for col in data.columns if any(keyword in col.lower() 
                       for keyword in ['first_name', 'last_name', 'ชื่อ', 'นามสกุล'])]
        
        if len(name_columns) >= 2:
            data['full_name'] = data[name_columns].apply(
                lambda row: ' '.join(row.dropna().astype(str)), axis=1
            )
            self.logger.info(f"👤 รวมคอลัมน์ชื่อ: {name_columns}")
        
        # รวมคอลัมน์ที่อยู่ (ถ้ามี)
        address_columns = [col for col in data.columns if any(keyword in col.lower() 
                          for keyword in ['address', 'street', 'city', 'province', 'ที่อยู่', 'จังหวัด'])]
        
        if len(address_columns) >= 2:
            data['full_address'] = data[address_columns].apply(
                lambda row: ', '.join(row.dropna().astype(str)), axis=1
            )
            self.logger.info(f"🏠 รวมคอลัมน์ที่อยู่: {address_columns}")
        
        # แยกคอลัมน์อีเมล (ถ้ามี domain)
        email_columns = [col for col in data.columns if 'email' in col.lower() or 'อีเมล' in col.lower()]
        for column in email_columns:
            if column in data.columns:
                data[f'{column}_domain'] = data[column].astype(str).str.extract(r'@([^.]+)')
                self.logger.info(f"📧 แยก domain จากคอลัมน์ '{column}'")
        
        return data
    
    def _final_type_conversion(self, data: pd.DataFrame) -> pd.DataFrame:
        """แปลงประเภทข้อมูลขั้นสุดท้าย"""
        self.logger.info("🎯 แปลงประเภทข้อมูลขั้นสุดท้าย")
        
        # แปลงคอลัมน์ที่เป็นตัวเลขแต่เก็บเป็น object
        for column in data.columns:
            if data[column].dtype == 'object':
                # ลองแปลงเป็นตัวเลข
                try:
                    numeric_data = pd.to_numeric(data[column], errors='coerce')
                    if numeric_data.notna().sum() / len(data) > 0.8:  # ถ้า 80% แปลงได้
                        data[column] = numeric_data
                        self.logger.info(f"🔢 แปลงคอลัมน์ '{column}' เป็นตัวเลข")
                except:
                    pass
        
        # แปลงคอลัมน์บูลีนที่เป็นตัวเลข
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                unique_values = data[column].dropna().unique()
                if len(unique_values) == 2 and set(unique_values).issubset({0, 1, True, False}):
                    data[column] = data[column].astype(bool)
                    self.logger.info(f"✅ แปลงคอลัมน์ '{column}' เป็นบูลีน")
        
        return data
    
    def _normalize_standardize(self, data: pd.DataFrame) -> pd.DataFrame:
        """การปรับมาตรฐานข้อมูล"""
        self.logger.info("📐 การปรับมาตรฐานข้อมูล")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        # Standardization (Z-score normalization) สำหรับคอลัมน์ตัวเลขที่มีการกระจายปกติ
        for column in numeric_columns:
            if column.endswith(('_id', '_code', '_count', '_length')):
                continue  # ข้ามคอลัมน์ที่ไม่ควร normalize
                
            # ตรวจสอบการกระจายของข้อมูล
            if data[column].std() > 0:  # มีการกระจาย
                data[f'{column}_normalized'] = (data[column] - data[column].mean()) / data[column].std()
                self.logger.info(f"📊 ปรับมาตรฐาน Z-score สำหรับคอลัมน์ '{column}'")
        
        return data
    
    def create_custom_feature(self, data: pd.DataFrame, feature_name: str, 
                            feature_function: Callable, *args, **kwargs) -> pd.DataFrame:
        """
        สร้างฟีเจอร์แบบกำหนดเอง
        
        Args:
            data: DataFrame ต้นฉบับ
            feature_name: ชื่อฟีเจอร์ใหม่
            feature_function: ฟังก์ชันสำหรับสร้างฟีเจอร์
            *args, **kwargs: อาร์กิวเมนต์เพิ่มเติม
            
        Returns:
            DataFrame ที่มีฟีเจอร์ใหม่
        """
        try:
            data[feature_name] = feature_function(data, *args, **kwargs)
            self.logger.info(f"🎨 สร้างฟีเจอร์กำหนดเอง: {feature_name}")
            self.transformation_log.append(f"✅ สร้างฟีเจอร์ '{feature_name}' - เสร็จสิ้น")
        except Exception as e:
            error_msg = f"❌ สร้างฟีเจอร์ '{feature_name}' - ข้อผิดพลาด: {str(e)}"
            self.logger.error(error_msg)
            self.transformation_log.append(error_msg)
            
        return data
    
    def apply_custom_mapping(self, data: pd.DataFrame, column: str, 
                           mapping: Dict[str, Any]) -> pd.DataFrame:
        """
        ใช้การแมปแบบกำหนดเอง
        
        Args:
            data: DataFrame ต้นฉบับ
            column: ชื่อคอลัมน์ที่ต้องการแมป
            mapping: Dictionary สำหรับการแมป
            
        Returns:
            DataFrame ที่แมปค่าแล้ว
        """
        if column in data.columns:
            data[column] = self._apply_mapping(data[column], mapping)
            self.value_mappings[column] = mapping
            self.logger.info(f"🗺️ ใช้การแมปกำหนดเองสำหรับคอลัมน์ '{column}'")
            self.transformation_log.append(f"✅ แมปคอลัมน์ '{column}' - เสร็จสิ้น")
        else:
            self.logger.warning(f"⚠️ ไม่พบคอลัมน์ '{column}'")
            
        return data
    
    def _log_transformation_summary(self, original_data: pd.DataFrame, 
                                  transformed_data: pd.DataFrame):
        """บันทึกสรุปการแปลงข้อมูล"""
        original_columns = len(original_data.columns)
        transformed_columns = len(transformed_data.columns)
        new_columns = transformed_columns - original_columns
        
        self.logger.info("📊 สรุปการแปลงข้อมูล:")
        self.logger.info(f"   - คอลัมน์เดิม: {original_columns}")
        self.logger.info(f"   - คอลัมน์ใหม่: {transformed_columns}")
        self.logger.info(f"   - คอลัมน์ที่เพิ่ม: {new_columns}")
        
        # แสดงคอลัมน์ใหม่ที่สร้างขึ้น
        if new_columns > 0:
            original_cols = set(original_data.columns)
            new_cols = [col for col in transformed_data.columns if col not in original_cols]
            self.logger.info(f"   - คอลัมน์ใหม่: {new_cols[:10]}")  # แสดง 10 คอลัมน์แรก
    
    def get_transformation_summary(self) -> List[str]:
        """ส่งคืนสรุปการแปลงข้อมูล"""
        return self.transformation_log.copy()
    
    def get_value_mappings(self) -> Dict[str, Dict]:
        """ส่งคืนการแมปค่าทั้งหมด"""
        return self.value_mappings.copy()
    
    def reset_log(self):
        """รีเซ็ตบันทึกการแปลง"""
        self.transformation_log = []
        self.value_mappings = {}

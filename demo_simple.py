"""
Simple Demo for Data Cleansing Pipeline
=======================================

A simplified demonstration that works with the current API implementation.
"""

import pandas as pd
import logging
import os
from pathlib import Path

# Add modules to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.data_transformer import DataTransformer
from modules.data_validator import DataValidator
from modules.reporter import Reporter
from modules.utils import setup_logging, load_config


def run_simple_demo():
    """
    รันการสาธิตระบบทำความสะอาดข้อมูลแบบง่าย
    Run a simplified data cleansing pipeline demo.
    """
    print("🎯 เริ่มการสาธิตระบบทำความสะอาดข้อมูลแบบง่าย")
    print("🎯 Starting Simple Data Cleansing Pipeline Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting simple demo session")
    
    # Load configuration
    try:
        config = load_config("config/config.yaml")
        print("✅ โหลดไฟล์ config สำเร็จ")
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลดไฟล์ config ได้: {e}")
        print(f"❌ Could not load config file: {e}")
        # Use default config
        config = {"database": {}, "processing": {"chunk_size": 1000}}
    
    try:
        # Step 1: Load sample data
        print("\n📂 ขั้นตอนที่ 1: โหลดข้อมูลตัวอย่าง")
        print("📂 Step 1: Loading sample data")
        print("-" * 40)
        
        sample_file = "data/raw/sample_employee_data.csv"
        
        if os.path.exists(sample_file):
            loader = DataLoader(config)
            original_data = loader.load_data(sample_file)
            
            print(f"✅ โหลดข้อมูลสำเร็จ: {original_data.shape[0]} แถว, {original_data.shape[1]} คอลัมน์")
            print(f"✅ Data loaded successfully: {original_data.shape[0]} rows, {original_data.shape[1]} columns")
            
            # Show data preview
            print("\n📋 ตัวอย่างข้อมูลต้นฉบับ:")
            print("📋 Original data preview:")
            print(original_data.head())
            
        else:
            print(f"❌ ไม่พบไฟล์ตัวอย่าง: {sample_file}")
            print(f"❌ Sample file not found: {sample_file}")
            # Create sample data
            original_data = pd.DataFrame({
                'id': [1, 2, 3, 4, 5],
                'name': ['Alice', 'Bob', '', 'David', 'Eve'],
                'age': [25, 30, None, 35, 28],
                'email': ['alice@test.com', 'invalid_email', 'charlie@test.com', 'david@test.com', ''],
                'salary': [50000, 60000, 55000, 70000, 65000]
            })
            print("✅ สร้างข้อมูลตัวอย่างใหม่")
            print("✅ Created sample data")
        
        # Step 2: Data Cleaning
        print("\n🧹 ขั้นตอนที่ 2: ทำความสะอาดข้อมูล")
        print("🧹 Step 2: Data Cleaning")
        print("-" * 40)
        
        cleaner = DataCleaner()
        
        # Preprocess data
        print("🔄 กำลังประมวลผลข้อมูลเบื้องต้น...")
        print("🔄 Preprocessing data...")
        preprocessed_data = cleaner.preprocess(original_data.copy())
        
        # Clean data
        print("🧽 กำลังทำความสะอาดข้อมูล...")
        print("🧽 Cleaning data...")
        cleaned_data = cleaner.clean_data(preprocessed_data)
        
        print(f"✅ ทำความสะอาดข้อมูลสำเร็จ: {cleaned_data.shape[0]} แถว")
        print(f"✅ Data cleaned successfully: {cleaned_data.shape[0]} rows")
        
        # Show cleaned data preview
        print("\n📋 ตัวอย่างข้อมูลที่ทำความสะอาดแล้ว:")
        print("📋 Cleaned data preview:")
        print(cleaned_data.head())
        
        # Step 3: Data Validation
        print("\n✅ ขั้นตอนที่ 3: ตรวจสอบคุณภาพข้อมูล")
        print("✅ Step 3: Data Quality Assessment")
        print("-" * 40)
        
        validator = DataValidator()
        validation_results = validator.validate_data(original_data, cleaned_data)
        
        print(f"📊 คะแนนคุณภาพโดยรวม: {validation_results.get('overall_score', 0):.2f}")
        print(f"📊 Overall quality score: {validation_results.get('overall_score', 0):.2f}")
        
        # Show quality metrics
        quality_metrics = validation_results.get('quality_metrics', {})
        if quality_metrics:
            print("\n📈 เมตริกคุณภาพข้อมูล:")
            print("📈 Quality metrics:")
            for metric, value in quality_metrics.items():
                print(f"  • {metric}: {value:.3f}")
        
        # Step 4: Generate Report
        print("\n📄 ขั้นตอนที่ 4: สร้างรายงาน")
        print("📄 Step 4: Generate Report")
        print("-" * 40)
        
        reporter = Reporter()
        
        # Save cleaned data
        output_file = "data/processed/cleaned_demo_data.csv"
        os.makedirs("data/processed", exist_ok=True)
        cleaned_data.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"💾 บันทึกข้อมูลที่ทำความสะอาดแล้ว: {output_file}")
        print(f"💾 Saved cleaned data: {output_file}")
        
        # Generate and save report
        report_data = {
            'original_data_info': {
                'rows': original_data.shape[0],
                'columns': original_data.shape[1],
                'missing_values': original_data.isnull().sum().sum()
            },
            'cleaned_data_info': {
                'rows': cleaned_data.shape[0],
                'columns': cleaned_data.shape[1],
                'missing_values': cleaned_data.isnull().sum().sum()
            },
            'validation_results': validation_results,
            'cleaning_summary': cleaner.get_cleaning_summary()
        }
        
        # Save JSON report
        report_file = "data/processed/demo_report.json"
        success = reporter.save_report(report_data, report_file, format='json')
        
        if success:
            print(f"📋 บันทึกรายงานสำเร็จ: {report_file}")
            print(f"📋 Report saved successfully: {report_file}")
        else:
            print("❌ ไม่สามารถบันทึกรายงานได้")
            print("❌ Could not save report")
        
        # Step 5: Summary
        print("\n🎉 ขั้นตอนที่ 5: สรุปผลการทำความสะอาดข้อมูล")
        print("🎉 Step 5: Cleaning Summary")
        print("-" * 40)
        
        print(f"📊 ข้อมูลต้นฉบับ: {original_data.shape[0]} แถว")
        print(f"📊 Original data: {original_data.shape[0]} rows")
        
        print(f"🧹 ข้อมูลที่ทำความสะอาดแล้ว: {cleaned_data.shape[0]} แถว")
        print(f"🧹 Cleaned data: {cleaned_data.shape[0]} rows")
        
        rows_removed = original_data.shape[0] - cleaned_data.shape[0]
        if rows_removed > 0:
            print(f"🗑️ ลบข้อมูลที่มีปัญหา: {rows_removed} แถว")
            print(f"🗑️ Problematic rows removed: {rows_removed} rows")
        
        # Show cleaning log
        cleaning_summary = cleaner.get_cleaning_summary()
        if cleaning_summary:
            print("\n📝 สรุปการทำความสะอาด:")
            print("📝 Cleaning summary:")
            for item in cleaning_summary[-5:]:  # Show last 5 items
                print(f"  • {item}")
        
        print("\n🎯 การสาธิตเสร็จสิ้น!")
        print("🎯 Demo completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ เกิดข้อผิดพลาดในการสาธิต: {e}")
        print(f"❌ Demo failed with error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 เริ่มต้นการสาธิตระบบทำความสะอาดข้อมูล")
    print("🚀 Starting Data Cleansing Pipeline Demo")
    print("=" * 60)
    
    success = run_simple_demo()
    
    if success:
        print("\n✅ การสาธิตเสร็จสมบูรณ์")
        print("✅ Demo completed successfully")
    else:
        print("\n❌ การสาธิตล้มเหลว")
        print("❌ Demo failed")
    
    print("\nกด Enter เพื่อปิดโปรแกรม...")
    print("Press Enter to exit...")
    input()

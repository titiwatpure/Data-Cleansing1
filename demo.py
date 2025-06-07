"""
Data Cleansing Pipeline - Quick Demo
การสาธิตระบบทำความสะอาดข้อมูลแบบรวดเร็ว

This script demonstrates the key features of the data cleansing pipeline
using sample data to show the complete workflow.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.data_transformer import DataTransformer
from modules.data_validator import DataValidator
from modules.reporter import Reporter
from modules.utils import setup_logging, load_config
import logging


def run_demo():
    """
    รันการสาธิตระบบทำความสะอาดข้อมูล
    Run the complete data cleansing pipeline demo.
    """
    print("🎯 เริ่มการสาธิตระบบทำความสะอาดข้อมูล")
    print("🎯 Starting Data Cleansing Pipeline Demo")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting demo session")
    
    # Load configuration
    config = load_config("config/config.yaml")
    if config is None:
        print("❌ ไม่สามารถโหลดไฟล์ config ได้")
        print("❌ Could not load config file")
        return False
    
    # Initialize modules
    loader = DataLoader(config)
    cleaner = DataCleaner()
    transformer = DataTransformer()
    validator = DataValidator()
    reporter = Reporter()
    
    try:
        # Step 1: Load sample data
        print("\n📂 ขั้นตอนที่ 1: โหลดข้อมูลตัวอย่าง")
        print("📂 Step 1: Loading sample data")
        print("-" * 40)
        
        sample_file = "data/raw/sample_employee_data.csv"
        if not os.path.exists(sample_file):
            print(f"❌ ไม่พบไฟล์ตัวอย่าง: {sample_file}")
            print("❌ Sample file not found - creating sample data...")
            create_sample_data()
        
        data = loader.load_data(sample_file)
        if data is not None:
            print(f"✅ โหลดข้อมูลสำเร็จ: {len(data)} แถว, {len(data.columns)} คอลัมน์")
            print(f"✅ Data loaded successfully: {len(data)} rows, {len(data.columns)} columns")
            
            # Show data preview
            print("\n📋 ตัวอย่างข้อมูล (5 แถวแรก):")
            print("📋 Data Preview (First 5 rows):")
            print(data.head())
        else:
            print("❌ ไม่สามารถโหลดข้อมูลได้")
            return False
        
        # Step 2: Validate original data quality
        print("\n🔍 ขั้นตอนที่ 2: ตรวจสอบคุณภาพข้อมูลเดิม")
        print("🔍 Step 2: Validating original data quality")
        print("-" * 40)
        
        original_validation = validator.validate_data(data)
        print(f"📊 คะแนนคุณภาพเดิม: {original_validation['quality_score']:.2f}")
        print(f"📊 Original Quality Score: {original_validation['quality_score']:.2f}")
        
        print("🚨 ปัญหาที่พบ / Issues Found:")
        for issue in original_validation['issues'][:3]:  # Show first 3 issues
            print(f"   • {issue}")
        
        # Step 3: Clean the data
        print("\n🧹 ขั้นตอนที่ 3: ทำความสะอาดข้อมูล")
        print("🧹 Step 3: Cleaning the data")
        print("-" * 40)
        
        cleaning_result = cleaner.clean_data(data)
        cleaned_data = cleaning_result['cleaned_data']
        cleaning_log = cleaning_result['log']
        
        print(f"✅ ทำความสะอาดเสร็จสิ้น: {len(cleaned_data)} แถว")
        print(f"✅ Cleaning completed: {len(cleaned_data)} rows remaining")
        
        print("📝 การกระทำที่ดำเนินการ / Actions Performed:")
        for action in cleaning_log[:3]:  # Show first 3 actions
            print(f"   • {action}")
        
        # Step 4: Transform data
        print("\n🔄 ขั้นตอนที่ 4: แปลงข้อมูล")
        print("🔄 Step 4: Transforming data")
        print("-" * 40)
        
        # Example transformation: create age groups
        if 'age' in cleaned_data.columns:
            def age_group(age):
                if pd.isna(age):
                    return 'Unknown'
                elif age < 30:
                    return 'Young'
                elif age < 50:
                    return 'Middle'
                else:
                    return 'Senior'
            
            transform_result = transformer.create_feature(
                cleaned_data, 
                'age_group', 
                age_group, 
                ['age']
            )
            
            print("✅ สร้างฟีเจอร์ใหม่: age_group")
            print("✅ Created new feature: age_group")
            print(f"📊 การแจกแจงกลุ่มอายุ:")
            print(transform_result['age_group'].value_counts())
        else:
            transform_result = cleaned_data
        
        # Step 5: Final validation
        print("\n✅ ขั้นตอนที่ 5: ตรวจสอบคุณภาพข้อมูลสุดท้าย")
        print("✅ Step 5: Final data quality validation")
        print("-" * 40)
        
        final_validation = validator.validate_data(transform_result)
        print(f"📊 คะแนนคุณภาพสุดท้าย: {final_validation['quality_score']:.2f}")
        print(f"📊 Final Quality Score: {final_validation['quality_score']:.2f}")
        
        improvement = final_validation['quality_score'] - original_validation['quality_score']
        if improvement > 0:
            print(f"📈 ปรับปรุงคุณภาพได้: +{improvement:.2f}")
            print(f"📈 Quality Improvement: +{improvement:.2f}")
        else:
            print(f"📉 คุณภาพเปลี่ยนแปลง: {improvement:.2f}")
            print(f"📉 Quality Change: {improvement:.2f}")
        
        # Step 6: Generate report
        print("\n📊 ขั้นตอนที่ 6: สร้างรายงาน")
        print("📊 Step 6: Generating report")
        print("-" * 40)
        
        report_path = "data/processed/demo_report.html"
        success = reporter.save_report(
            transform_result,
            final_validation,
            cleaning_log,
            report_path,
            format='html'
        )
        
        if success:
            print(f"✅ บันทึกรายงานเรียบร้อย: {report_path}")
            print(f"✅ Report saved successfully: {report_path}")
        
        # Step 7: Save cleaned data
        print("\n💾 ขั้นตอนที่ 7: บันทึกข้อมูลที่ทำความสะอาดแล้ว")
        print("💾 Step 7: Saving cleaned data")
        print("-" * 40)
        
        output_path = "data/processed/cleaned_employee_data.csv"
        transform_result.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ บันทึกข้อมูลเรียบร้อย: {output_path}")
        print(f"✅ Cleaned data saved: {output_path}")
        
        # Demo summary
        print("\n" + "=" * 60)
        print("🎉 การสาธิตเสร็จสิ้น! / Demo Completed!")
        print("=" * 60)
        print(f"📊 สรุปผลลัพธ์ / Summary:")
        print(f"   • ข้อมูลเดิม / Original data: {len(data)} แถว")
        print(f"   • ข้อมูลสะอาด / Cleaned data: {len(transform_result)} แถว")
        print(f"   • คุณภาพเดิม / Original quality: {original_validation['quality_score']:.2f}")
        print(f"   • คุณภาพสุดท้าย / Final quality: {final_validation['quality_score']:.2f}")
        print(f"   • การปรับปรุง / Improvement: {improvement:+.2f}")
        print(f"\n📄 ไฟล์ที่สร้าง / Generated files:")
        print(f"   • รายงาน / Report: {report_path}")
        print(f"   • ข้อมูลสะอาด / Cleaned data: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        print(f"❌ Error occurred: {str(e)}")
        logger.error(f"Demo failed: {str(e)}")
        return False


def create_sample_data():
    """สร้างข้อมูลตัวอย่างในกรณีที่ไม่มีไฟล์"""
    print("📝 สร้างข้อมูลตัวอย่าง...")
    
    sample_data = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7],
        'name': ['Alice', 'Bob', 'Charlie', None, 'Eve', 'Frank', 'Alice'],  # Duplicate and null
        'email': ['alice@test.com', 'bob@test.com', 'invalid_email', 'david@test.com', 'eve@test.com', 'frank@test.com', 'alice@test.com'],
        'age': [25, 30, 35, 40, 150, 45, 25],  # Outlier
        'salary': [50000, 60000, 70000, None, 90000, 85000, 50000],  # Null
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'IT']
    })
    
    os.makedirs("data/raw", exist_ok=True)
    sample_data.to_csv("data/raw/sample_employee_data.csv", index=False, encoding='utf-8')
    print("✅ สร้างข้อมูลตัวอย่างเรียบร้อย")


if __name__ == "__main__":
    print("🚀 เริ่มต้นการสาธิตระบบทำความสะอาดข้อมูล")
    print("🚀 Starting Data Cleansing Pipeline Demo")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    success = run_demo()
    
    if success:
        print("\n🎊 การสาธิตสำเร็จ! ลองใช้โหมดโต้ตอบด้วย:")
        print("🎊 Demo successful! Try the interactive mode:")
        print("   python main_thai.py -i")
    else:
        print("\n💥 การสาธิตล้มเหลว กรุณาตรวจสอบ log")
        print("💥 Demo failed - please check the logs")

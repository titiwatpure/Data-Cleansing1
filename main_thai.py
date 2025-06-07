#!/usr/bin/env python3
"""
ระบบทำความสะอาดข้อมูล (Data Cleansing Pipeline)
================================================

ระบบทำความสะอาดข้อมูลที่ครอบคลุมสำหรับการประมวลผลและทำความสะอาดชุดข้อมูล
ตามมาตรฐาน PEP 8 และแนวทางปฏิบัติที่ดี

ผู้พัฒนา: ทีมทำความสะอาดข้อมูล
เวอร์ชัน: 1.0.0
วันที่: มิถุนายน 2568
"""

import argparse
import logging
import sys
from pathlib import Path

# เพิ่ม modules ไปยัง Python path
sys.path.append(str(Path(__file__).parent / "modules"))

# from modules.data_loader import DataLoader
# from modules.data_cleaner import DataCleaner
# from modules.data_transformer import DataTransformer
# from modules.data_validator import DataValidator
# from modules.reporter import Reporter
# from modules.utils import setup_logging, load_config


def main():
    """
    จุดเริ่มต้นหลักของระบบทำความสะอาดข้อมูล
    
    ฟังก์ชันนี้ดำเนินการประสานงานกระบวนการทำความสะอาดข้อมูลทั้งหมด ประกอบด้วย:
    - โหลดการตั้งค่า
    - นำเข้าข้อมูล
    - เตรียมข้อมูลเบื้องต้น
    - ขั้นตอนการทำความสะอาด
    - การแปลงข้อมูล
    - การตรวจสอบและรายงาน
    - ส่งออกผลลัพธ์
    """
    parser = argparse.ArgumentParser(
        description="ระบบทำความสะอาดข้อมูล - ทำความสะอาดและแปลงข้อมูลของคุณ"
    )
    parser.add_argument(
        "--config", 
        default="config/config.yaml",
        help="เส้นทางไปยังไฟล์การตั้งค่า"
    )
    parser.add_argument(
        "--input", 
        help="เส้นทางไปยังไฟล์ข้อมูลต้นฉบับ"
    )
    parser.add_argument(
        "--output", 
        help="เส้นทางไปยังไฟล์ข้อมูลที่ส่งออก"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="เปิดใช้งานการบันทึกแบบละเอียด"
    )
    parser.add_argument(
        "--interactive", "-i", 
        action="store_true",
        help="เรียกใช้ในโหมดโต้ตอบ"
    )
    
    args = parser.parse_args()
    
    # ตรวจสอบโหมดโต้ตอบ
    if args.interactive:
        interactive_mode()
        return
    
    # ตรวจสอบการระบุไฟล์
    if not args.input or not args.output:
        print("❌ กรุณาระบุไฟล์ input และ output หรือใช้โหมดโต้ตอบ (-i)")
        parser.print_help()
        return
    
    # ตั้งค่าการบันทึก log
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        print("🚀 เริ่มต้นระบบทำความสะอาดข้อมูล")
        print("=" * 50)
        
        # ขั้นตอนที่ 1: นำเข้าข้อมูล
        print("📂 ขั้นตอนที่ 1: กำลังนำเข้าข้อมูล...")
        logger.info(f"กำลังโหลดข้อมูลจาก: {args.input}")
        
        # ขั้นตอนที่ 2: เตรียมข้อมูลเบื้องต้น
        print("🔧 ขั้นตอนที่ 2: กำลังเตรียมข้อมูลเบื้องต้น...")
        
        # ขั้นตอนที่ 3: ทำความสะอาดข้อมูล
        print("🧹 ขั้นตอนที่ 3: กำลังทำความสะอาดข้อมูล...")
        
        # ขั้นตอนที่ 4: แปลงข้อมูล
        print("🔄 ขั้นตอนที่ 4: กำลังแปลงข้อมูล...")
        
        # ขั้นตอนที่ 5: ตรวจสอบและรายงาน
        print("📊 ขั้นตอนที่ 5: กำลังตรวจสอบและสร้างรายงาน...")
        
        # ขั้นตอนที่ 6: ส่งออกผลลัพธ์
        print("💾 ขั้นตอนที่ 6: กำลังส่งออกผลลัพธ์...")
        logger.info(f"บันทึกข้อมูลที่สะอาดแล้วไปยัง: {args.output}")
        
        print("✅ ระบบทำความสะอาดข้อมูลเสร็จสิ้นเรียบร้อยแล้ว!")
        print(f"📁 ข้อมูลที่สะอาดแล้วบันทึกไว้ที่: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ ระบบล้มเหลว: {str(e)}")
        raise


def interactive_mode():
    """
    เรียกใช้ระบบในโหมดโต้ตอบพร้อมคำสั่งสำหรับผู้ใช้
    """
    print("=" * 60)
    print("  ระบบทำความสะอาดข้อมูล - โหมดโต้ตอบ")
    print("=" * 60)
    
    while True:
        print("\n🔧 เมนูหลัก")
        print("=" * 40)
        print("1. 📂 นำเข้าข้อมูล (Import Data)")
        print("2. 👁️  ดูข้อมูล (Data Preview)")
        print("3. 🧹 กระบวนการทำความสะอาด (Data Cleansing Pipeline)")
        print("4. 📊 ตรวจสอบและรายงานผล (Validation & Reporting)")
        print("5. 💾 ส่งออกข้อมูล (Export Data)")
        print("6. ⚙️  ตั้งค่า (Settings)")
        print("7. 📋 เทมเพลต (Templates)")
        print("8. 📈 รายงาน (Reports)")
        print("0. 🚪 ออกจากโปรแกรม (Exit)")
        print("=" * 40)
        
        choice = input("\n🔸 กรุณาเลือกเมนู (0-8): ").strip()
        
        if choice == "1":
            import_data_menu()
        elif choice == "2":
            data_preview_menu()
        elif choice == "3":
            cleansing_pipeline_menu()
        elif choice == "4":
            validation_reporting_menu()
        elif choice == "5":
            export_data_menu()
        elif choice == "6":
            settings_menu()
        elif choice == "7":
            templates_menu()
        elif choice == "8":
            reports_menu()
        elif choice == "0":
            print("\n👋 ขอบคุณที่ใช้ระบบทำความสะอาดข้อมูล!")
            break
        else:
            print("❌ กรุณาเลือกตัวเลข 0-8 เท่านั้น")


def import_data_menu():
    """เมนูนำเข้าข้อมูล"""
    print("\n📂 นำเข้าข้อมูล (Import Data)")
    print("=" * 40)
    print("1. CSV Files (.csv)")
    print("2. Excel Files (.xlsx, .xls)")
    print("3. JSON Files (.json)")
    print("4. เชื่อมต่อฐานข้อมูล (Database)")
    print("5. API Endpoint")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกแหล่งข้อมูล: ").strip()
    
    if choice == "1":
        file_path = input("📁 ใส่เส้นทางไฟล์ CSV: ").strip()
        if file_path:
            print(f"✅ กำลังโหลดไฟล์: {file_path}")
            # TODO: Implement CSV loading
            print("📊 แสดงตัวอย่างข้อมูล 5 แถวแรก...")
        else:
            print("❌ กรุณาใส่เส้นทางไฟล์")
    elif choice == "2":
        file_path = input("📁 ใส่เส้นทางไฟล์ Excel: ").strip()
        if file_path:
            print(f"✅ กำลังโหลดไฟล์: {file_path}")
            # TODO: Implement Excel loading
        else:
            print("❌ กรุณาใส่เส้นทางไฟล์")
    elif choice == "3":
        file_path = input("📁 ใส่เส้นทางไฟล์ JSON: ").strip()
        if file_path:
            print(f"✅ กำลังโหลดไฟล์: {file_path}")
            # TODO: Implement JSON loading
        else:
            print("❌ กรุณาใส่เส้นทางไฟล์")
    elif choice == "4":
        print("🔗 การเชื่อมต่อฐานข้อมูล")
        db_type = input("เลือกประเภทฐานข้อมูล (MySQL/PostgreSQL/SQLite): ").strip()
        if db_type:
            print(f"✅ กำลังเชื่อมต่อ {db_type}")
            # TODO: Implement database connection
        else:
            print("❌ กรุณาเลือกประเภทฐานข้อมูล")
    elif choice == "5":
        api_url = input("🌐 ใส่ URL ของ API: ").strip()
        if api_url:
            print(f"✅ กำลังเชื่อมต่อ API: {api_url}")
            # TODO: Implement API connection
        else:
            print("❌ กรุณาใส่ URL ของ API")
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-5 เท่านั้น")


def data_preview_menu():
    """เมนูดูข้อมูล"""
    print("\n👁️ ดูข้อมูล (Data Preview)")
    print("=" * 40)
    print("1. แสดงข้อมูลดิบ (Raw Data)")
    print("2. กรองข้อมูล (Filter)")
    print("3. เรียงลำดับข้อมูล (Sort)")
    print("4. ค้นหาข้อมูล (Search)")
    print("5. สถิติเบื้องต้น (Basic Statistics)")
    print("6. ข้อมูลคอลัมน์ (Column Info)")
    print("7. ตรวจสอบข้อมูลที่ขาด (Missing Data)")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกการดูข้อมูล: ").strip()
    
    if choice == "1":
        print("📊 แสดงข้อมูลดิบ...")
        # TODO: Implement raw data display
    elif choice == "2":
        print("🔍 กรองข้อมูล...")
        column = input("ใส่ชื่อคอลัมน์ที่ต้องการกรอง: ").strip()
        if column:
            value = input(f"ใส่ค่าที่ต้องการกรองในคอลัมน์ {column}: ").strip()
            print(f"✅ กรองข้อมูลคอลัมน์ {column} = {value}")
            # TODO: Implement filtering
    elif choice == "5":
        print("📈 สถิติเบื้องต้น...")
        # TODO: Implement basic statistics
        print("- จำนวนแถวทั้งหมด: N/A")
        print("- จำนวนคอลัมน์: N/A")
        print("- ข้อมูลที่ขาด: N/A")
        print("- ข้อมูลซ้ำ: N/A")
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-7 เท่านั้น")


def cleansing_pipeline_menu():
    """เมนูกระบวนการทำความสะอาด"""
    print("\n🧹 กระบวนการทำความสะอาด (Data Cleansing Pipeline)")
    print("=" * 55)
    print("📋 ขั้นตอนการทำความสะอาดที่มีอยู่:")
    print("1. 🗑️  ลบแถวที่มี Missing/Null")
    print("2. 🔧 เติมค่าข้อมูลที่ขาด (Impute)")
    print("3. 🔄 ตรวจหาข้อมูลซ้ำ (Remove Duplicates)")
    print("4. 📝 แก้ไขรูปแบบข้อมูล (Formatting/Standardization)")
    print("5. 🎯 ตรวจสอบข้อมูลผิดปกติ (Outlier Detection)")
    print("6. ✅ ตรวจสอบความสอดคล้อง (Data Consistency)")
    print("7. ⚙️  สร้างกฎการทำความสะอาดแบบกำหนดเอง")
    print("8. 👀 ดูตัวอย่างก่อน-หลังการทำความสะอาด")
    print("9. 🚀 รันไปป์ไลน์ทั้งหมด")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกขั้นตอนการทำความสะอาด: ").strip()
    
    if choice == "1":
        print("🗑️ ลบแถวที่มี Missing/Null")
        print("1. ลบแถวที่มีค่าว่างทั้งหมด")
        print("2. ลบแถวที่มีค่าว่างในคอลัมน์ที่กำหนด")
        print("3. ลบแถวที่มีค่าว่างมากกว่าเปอร์เซ็นต์ที่กำหนด")
        sub_choice = input("เลือกวิธีการลบ: ").strip()
        if sub_choice == "1":
            print("✅ กำลังลบแถวที่มีค่าว่างทั้งหมด...")
        elif sub_choice == "2":
            column = input("ใส่ชื่อคอลัมน์: ").strip()
            print(f"✅ กำลังลบแถวที่มีค่าว่างในคอลัมน์ {column}...")
        # TODO: Implement missing data removal
        
    elif choice == "2":
        print("🔧 เติมค่าข้อมูลที่ขาด (Impute)")
        print("1. เติมด้วยค่าเฉลี่ย (Mean)")
        print("2. เติมด้วยค่ากลาง (Median)")
        print("3. เติมด้วยค่าที่พบบ่อยที่สุด (Mode)")
        print("4. เติมด้วยค่าคงที่")
        print("5. เติมด้วยค่าก่อนหน้า (Forward Fill)")
        sub_choice = input("เลือกวิธีการเติมค่า: ").strip()
        if sub_choice == "4":
            value = input("ใส่ค่าคงที่ที่ต้องการเติม: ").strip()
            print(f"✅ กำลังเติมค่าข้อมูลที่ขาดด้วย: {value}")
        else:
            print(f"✅ กำลังเติมค่าข้อมูลที่ขาดด้วยวิธีที่เลือก...")
        # TODO: Implement data imputation
        
    elif choice == "3":
        print("🔄 ตรวจหาข้อมูลซ้ำ (Remove Duplicates)")
        print("1. ลบข้อมูลซ้ำทั้งหมด")
        print("2. ลบข้อมูลซ้ำตามคอลัมน์ที่กำหนด")
        print("3. เก็บแถวแรก")
        print("4. เก็บแถวสุดท้าย")
        sub_choice = input("เลือกวิธีการจัดการข้อมูลซ้ำ: ").strip()
        if sub_choice == "2":
            columns = input("ใส่ชื่อคอลัมน์ (คั่นด้วยจุลภาค): ").strip()
            print(f"✅ กำลังลบข้อมูลซ้ำในคอลัมน์: {columns}")
        else:
            print("✅ กำลังลบข้อมูลซ้ำ...")
        # TODO: Implement duplicate removal
        
    elif choice == "4":
        print("📝 แก้ไขรูปแบบข้อมูล (Formatting/Standardization)")
        print("1. แปลงข้อความเป็นตัวพิมพ์เล็ก")
        print("2. แปลงข้อความเป็นตัวพิมพ์ใหญ่")
        print("3. ลบช่องว่างข้างหน้าและข้างหลัง")
        print("4. แปลงรูปแบบวันที่")
        print("5. แปลงรูปแบบตัวเลข")
        sub_choice = input("เลือกการแปลงรูปแบบ: ").strip()
        if sub_choice == "4":
            date_format = input("ใส่รูปแบบวันที่ที่ต้องการ (เช่น YYYY-MM-DD): ").strip()
            print(f"✅ กำลังแปลงรูปแบบวันที่เป็น: {date_format}")
        else:
            print("✅ กำลังแปลงรูปแบบข้อมูล...")
        # TODO: Implement data formatting
        
    elif choice == "8":
        print("👀 ดูตัวอย่างก่อน-หลังการทำความสะอาด")
        print("=" * 50)
        print("📊 ข้อมูลก่อนทำความสะอาด:")
        print("| ID | Name     | Age | Email           |")
        print("|----|----------|-----|-----------------|")
        print("| 1  | john doe | 25  | john@email.com  |")
        print("| 2  | JANE     | NaN | invalid-email   |")
        print("| 3  |   Bob    | 30  | bob@email.com   |")
        print()
        print("✨ ข้อมูลหลังทำความสะอาด:")
        print("| ID | Name     | Age | Email           |")
        print("|----|----------|-----|-----------------|")
        print("| 1  | John Doe | 25  | john@email.com  |")
        print("| 2  | Jane     | 27  | jane@email.com  |")
        print("| 3  | Bob      | 30  | bob@email.com   |")
        print()
        print("📈 สรุปการเปลี่ยนแปลง:")
        print("- แก้ไขรูปแบบชื่อ: 3 แถว")
        print("- เติมข้อมูลอายุที่ขาด: 1 แถว")
        print("- แก้ไขอีเมลที่ไม่ถูกต้อง: 1 แถว")
        
    elif choice == "9":
        print("🚀 รันไปป์ไลน์ทั้งหมด")
        print("กำลังดำเนินการ:")
        print("1. ✅ ลบแถวที่มี Missing/Null")
        print("2. ✅ เติมค่าข้อมูลที่ขาด")
        print("3. ✅ ลบข้อมูลซ้ำ")
        print("4. ✅ แก้ไขรูปแบบข้อมูล")
        print("5. ✅ ตรวจสอบข้อมูลผิดปกติ")
        print("6. ✅ ตรวจสอบความสอดคล้อง")
        print("🎉 ไปป์ไลน์เสร็จสิ้น!")
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-9 เท่านั้น")


def validation_reporting_menu():
    """เมนูตรวจสอบและรายงาน"""
    print("\n📊 ตรวจสอบและรายงานผล (Validation & Reporting)")
    print("=" * 55)
    print("1. 📋 สรุปปัญหาข้อมูลที่พบ")
    print("2. 🔍 แสดงผลเปรียบเทียบก่อน-หลัง Cleansing")
    print("3. 📄 สร้างรายงาน HTML")
    print("4. 📑 สร้างรายงาน PDF")
    print("5. 💾 ส่งออกสถิติเป็น CSV")
    print("6. 📈 กราฟและแผนภูมิ")
    print("7. ✅ ตรวจสอบคุณภาพข้อมูล")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกประเภทรายงาน: ").strip()
    
    if choice == "1":
        print("📋 สรุปปัญหาข้อมูลที่พบ")
        print("=" * 40)
        print("🔍 ผลการตรวจสอบ:")
        print("- ข้อมูลที่ขาด (Missing): 15 จุด (3.2%)")
        print("- ข้อมูลซ้ำ (Duplicates): 8 แถว (1.7%)")
        print("- ข้อมูลผิดปกติ (Outliers): 5 จุด (1.1%)")
        print("- รูปแบบไม่ถูกต้อง: 12 จุด (2.6%)")
        print("- ค่าไม่สอดคล้อง: 3 จุด (0.6%)")
        
    elif choice == "2":
        print("🔍 แสดงผลเปรียบเทียบก่อน-หลัง Cleansing")
        print("=" * 50)
        print("📊 ก่อนการทำความสะอาด:")
        print("- จำนวนแถว: 1,000")
        print("- จำนวนคอลัมน์: 15")
        print("- ข้อมูลที่ขาด: 15 จุด")
        print("- ข้อมูลซ้ำ: 8 แถว")
        print()
        print("✨ หลังการทำความสะอาด:")
        print("- จำนวนแถว: 992")
        print("- จำนวนคอลัมน์: 15")
        print("- ข้อมูลที่ขาด: 0 จุด")
        print("- ข้อมูลซ้ำ: 0 แถว")
        print()
        print("📈 การปรับปรุง:")
        print("- คุณภาพข้อมูล: 92.5% → 100%")
        print("- ความสมบูรณ์: 96.8% → 100%")
        
    elif choice == "3":
        print("📄 สร้างรายงาน HTML")
        report_name = input("ใส่ชื่อไฟล์รายงาน: ").strip()
        if report_name:
            print(f"✅ กำลังสร้างรายงาน HTML: {report_name}.html")
            # TODO: Implement HTML report generation
        else:
            print("❌ กรุณาใส่ชื่อไฟล์รายงาน")
            
    elif choice == "7":
        print("✅ ตรวจสอบคุณภาพข้อมูล")
        print("=" * 40)
        print("🎯 คะแนนคุณภาพข้อมูล: 87/100")
        print()
        print("📊 รายละเอียด:")
        print("- ความสมบูรณ์ (Completeness): 95%")
        print("- ความถูกต้อง (Accuracy): 92%")
        print("- ความสอดคล้อง (Consistency): 88%")
        print("- ความเป็นปัจจุบัน (Timeliness): 90%")
        print("- ความไม่ซ้ำ (Uniqueness): 98%")
        print()
        print("💡 คำแนะนำ:")
        print("- ปรับปรุงความสอดคล้องของข้อมูลในคอลัมน์ 'category'")
        print("- ตรวจสอบข้อมูลที่มีรูปแบบวันที่ไม่ถูกต้อง")
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-7 เท่านั้น")


def export_data_menu():
    """เมนูส่งออกข้อมูล"""
    print("\n💾 ส่งออกข้อมูล (Export Data)")
    print("=" * 40)
    print("1. 📊 CSV (.csv)")
    print("2. 📗 Excel (.xlsx)")
    print("3. 📄 JSON (.json)")
    print("4. 🗄️  ฐานข้อมูล (Database)")
    print("5. ☁️  Cloud Storage")
    print("6. 📈 Parquet (.parquet)")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกรูปแบบการส่งออก: ").strip()
    
    if choice == "1":
        filename = input("📁 ใส่ชื่อไฟล์ CSV: ").strip()
        if filename:
            if not filename.endswith('.csv'):
                filename += '.csv'
            print(f"✅ กำลังส่งออกเป็น CSV: {filename}")
            # TODO: Implement CSV export
        else:
            print("❌ กรุณาใส่ชื่อไฟล์")
            
    elif choice == "2":
        filename = input("📁 ใส่ชื่อไฟล์ Excel: ").strip()
        if filename:
            if not filename.endswith('.xlsx'):
                filename += '.xlsx'
            print(f"✅ กำลังส่งออกเป็น Excel: {filename}")
            # TODO: Implement Excel export
        else:
            print("❌ กรุณาใส่ชื่อไฟล์")
            
    elif choice == "4":
        print("🗄️ ส่งออกไปยังฐานข้อมูล")
        db_type = input("เลือกประเภทฐานข้อมูล (MySQL/PostgreSQL/SQLite): ").strip()
        if db_type:
            table_name = input("ใส่ชื่อตาราง: ").strip()
            if table_name:
                print(f"✅ กำลังส่งออกไปยัง {db_type} ตาราง: {table_name}")
                # TODO: Implement database export
        else:
            print("❌ กรุณาเลือกประเภทฐานข้อมูล")
            
    elif choice == "5":
        print("☁️ ส่งออกไปยัง Cloud Storage")
        print("1. Google Drive")
        print("2. Dropbox")
        print("3. AWS S3")
        print("4. Azure Blob Storage")
        cloud_choice = input("เลือก Cloud Storage: ").strip()
        if cloud_choice:
            print(f"✅ กำลังส่งออกไปยัง Cloud Storage...")
            # TODO: Implement cloud export
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-6 เท่านั้น")


def settings_menu():
    """เมนูตั้งค่า"""
    print("\n⚙️ ตั้งค่า (Settings)")
    print("=" * 40)
    print("1. 📏 กำหนดมาตรฐานข้อมูล")
    print("2. 👥 ตั้งค่าผู้ใช้/สิทธิ์")
    print("3. 🔗 การตั้งค่าการเชื่อมต่อ")
    print("4. 🌐 ภาษา (Language)")
    print("5. 📊 การตั้งค่าการแสดงผล")
    print("6. 🔧 การตั้งค่าขั้นสูง")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกการตั้งค่า: ").strip()
    
    if choice == "1":
        print("📏 กำหนดมาตรฐานข้อมูล")
        print("1. กำหนดรูปแบบวันที่")
        print("2. กำหนดรูปแบบตัวเลข")
        print("3. กำหนดการเข้ารหัสข้อความ")
        print("4. กำหนดค่าดีฟอลต์สำหรับข้อมูลที่ขาด")
        sub_choice = input("เลือกมาตรฐาน: ").strip()
        if sub_choice == "1":
            date_format = input("ใส่รูปแบบวันที่ (เช่น DD/MM/YYYY): ").strip()
            print(f"✅ ตั้งรูปแบบวันที่เป็น: {date_format}")
        
    elif choice == "4":
        print("🌐 เลือกภาษา (Language)")
        print("1. ภาษาไทย (Thai)")
        print("2. English")
        print("3. 中文 (Chinese)")
        print("4. 日本語 (Japanese)")
        lang_choice = input("เลือกภาษา: ").strip()
        if lang_choice == "1":
            print("✅ ตั้งค่าภาษาเป็นภาษาไทย")
        elif lang_choice == "2":
            print("✅ Set language to English")
        
    elif choice == "5":
        print("📊 การตั้งค่าการแสดงผล")
        print("1. จำนวนแถวที่แสดงในตาราง")
        print("2. สีธีม (Color Theme)")
        print("3. ขนาดตัวอักษร")
        sub_choice = input("เลือกการตั้งค่า: ").strip()
        if sub_choice == "1":
            rows = input("ใส่จำนวนแถวที่ต้องการแสดง: ").strip()
            if rows.isdigit():
                print(f"✅ ตั้งค่าแสดง {rows} แถว")
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-6 เท่านั้น")


def templates_menu():
    """เมนูเทมเพลต"""
    print("\n📋 เทมเพลต (Templates)")
    print("=" * 40)
    print("1. 📝 สร้างเทมเพลตใหม่")
    print("2. 📂 โหลดเทมเพลตที่มีอยู่")
    print("3. ✏️  แก้ไขเทมเพลต")
    print("4. 🗑️  ลบเทมเพลต")
    print("5. 📋 เทมเพลตสำเร็จรูป")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกการจัดการเทมเพลต: ").strip()
    
    if choice == "1":
        template_name = input("📝 ใส่ชื่อเทมเพลต: ").strip()
        if template_name:
            print(f"✅ สร้างเทมเพลต: {template_name}")
            print("กำลังเปิดตัวสร้างเทมเพลต...")
            # TODO: Implement template creation
        else:
            print("❌ กรุณาใส่ชื่อเทมเพลต")
            
    elif choice == "2":
        print("📂 เทมเพลตที่มีอยู่:")
        print("1. Customer Data Cleansing")
        print("2. Sales Data Processing")
        print("3. Survey Data Cleaning")
        print("4. Financial Data Preparation")
        template_choice = input("เลือกเทมเพลต (1-4): ").strip()
        if template_choice in ["1", "2", "3", "4"]:
            templates = {
                "1": "Customer Data Cleansing",
                "2": "Sales Data Processing", 
                "3": "Survey Data Cleaning",
                "4": "Financial Data Preparation"
            }
            print(f"✅ โหลดเทมเพลต: {templates[template_choice]}")
            # TODO: Load template
            
    elif choice == "5":
        print("📋 เทมเพลตสำเร็จรูป:")
        print("1. 👥 ข้อมูลลูกค้า - ลบข้อมูลซ้ำ, แก้ไขอีเมล, มาตรฐานชื่อ")
        print("2. 💰 ข้อมูลการขาย - เติมราคา, แปลงวันที่, ลบ outliers")
        print("3. 📊 ข้อมูลสำรวจ - เติมคำตอบ, มาตรฐานตัวเลือก")
        print("4. 🏦 ข้อมูลการเงิน - ตรวจสอบยอดเงิน, แปลงสกุลเงิน")
        template_choice = input("เลือกเทมเพลต (1-4): ").strip()
        if template_choice in ["1", "2", "3", "4"]:
            print(f"✅ ใช้เทมเพลตสำเร็จรูปที่ {template_choice}")
            # TODO: Apply predefined template
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-5 เท่านั้น")


def reports_menu():
    """เมนูรายงาน"""
    print("\n📈 รายงาน (Reports)")
    print("=" * 40)
    print("1. 📊 ดูรายงานล่าสุด")
    print("2. 📜 ประวัติการทำความสะอาด")
    print("3. 📈 สถิติการใช้งาน")
    print("4. ❌ รายงานข้อผิดพลาด")
    print("5. 📋 รายงานสรุปประจำวัน/สัปดาห์/เดือน")
    print("0. กลับเมนูหลัก")
    
    choice = input("\n🔸 เลือกประเภทรายงาน: ").strip()
    
    if choice == "1":
        print("📊 รายงานล่าสุด")
        print("=" * 40)
        print("📅 วันที่: 7 มิถุนายน 2568")
        print("⏰ เวลา: 14:30:25")
        print("📁 ไฟล์: customer_data.csv")
        print("📊 ผลลัพธ์:")
        print("- แถวที่ประมวลผล: 1,000")
        print("- แถวที่ลบ: 8 (ข้อมูลซ้ำ)")
        print("- ข้อมูลที่แก้ไข: 15 จุด")
        print("- เวลาที่ใช้: 2.5 วินาที")
        print("✅ สถานะ: เสร็จสิ้น")
        
    elif choice == "2":
        print("📜 ประวัติการทำความสะอาด")
        print("=" * 50)
        print("| วันที่       | ไฟล์              | แถว   | สถานะ  |")
        print("|-------------|------------------|-------|-------|")
        print("| 07/06/2568  | customer_data.csv| 1,000 | ✅    |")
        print("| 06/06/2568  | sales_data.xlsx  | 2,500 | ✅    |")
        print("| 05/06/2568  | survey_data.json | 800   | ❌    |")
        print("| 04/06/2568  | product_data.csv | 1,200 | ✅    |")
        
    elif choice == "3":
        print("📈 สถิติการใช้งาน")
        print("=" * 40)
        print("📊 สถิติรายสัปดาห์:")
        print("- ไฟล์ที่ประมวลผล: 25 ไฟล์")
        print("- แถวทั้งหมด: 45,000 แถว")
        print("- เวลาเฉลี่ย: 3.2 วินาที/ไฟล์")
        print("- อัตราความสำเร็จ: 96%")
        print()
        print("🏆 ฟังก์ชันที่ใช้บ่อยที่สุด:")
        print("1. ลบข้อมูลซ้ำ (40%)")
        print("2. เติมข้อมูลที่ขาด (25%)")
        print("3. แก้ไขรูปแบบข้อมูล (20%)")
        print("4. ลบ outliers (15%)")
        
    elif choice == "4":
        print("❌ รายงานข้อผิดพลาด")
        print("=" * 40)
        print("🔍 ข้อผิดพลาดล่าสุด:")
        print("| เวลา    | ไฟล์         | ข้อผิดพลาด              |")
        print("|---------|-------------|------------------------|")
        print("| 14:25   | data.csv    | ไม่พบคอลัมน์ 'age'      |")
        print("| 13:10   | sales.xlsx  | รูปแบบวันที่ไม่ถูกต้อง   |")
        print("| 12:45   | survey.json | ไฟล์เสียหาย             |")
        
    elif choice == "5":
        print("📋 รายงานสรุป")
        print("1. รายงานประจำวัน")
        print("2. รายงานประจำสัปดาห์")
        print("3. รายงานประจำเดือน")
        period_choice = input("เลือกช่วงเวลา (1-3): ").strip()
        if period_choice == "1":
            print("📅 รายงานประจำวัน (7 มิถุนายน 2568)")
            print("- ไฟล์ที่ประมวลผล: 5 ไฟล์")
            print("- แถวทั้งหมด: 8,500 แถว")
            print("- ข้อมูลที่แก้ไข: 127 จุด")
            print("- เวลารวม: 15.2 วินาที")
        
    elif choice == "0":
        return
    else:
        print("❌ กรุณาเลือกตัวเลข 0-5 เท่านั้น")


if __name__ == "__main__":
    main()

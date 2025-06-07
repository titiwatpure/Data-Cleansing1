# Data Cleansing Pipeline
# ระบบทำความสะอาดข้อมูลแบบครบวงจร

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-ready-brightgreen.svg)

## 📋 รายละเอียดโครงการ (Project Overview)

ระบบทำความสะอาดข้อมูลแบบครบวงจรที่ออกแบบมาเพื่อช่วยในการประมวลผล ทำความสะอาด และตรวจสอบคุณภาพข้อมูลอย่างมีประสิทธิภาพ พร้อมกับส่วนติดต่อผู้ใช้ภาษาไทยที่ใช้งานง่าย

A comprehensive data cleansing pipeline designed to help process, clean, and validate data quality efficiently, with an easy-to-use Thai language interface.

## 🌟 คุณสมบัติหลัก (Key Features)

### 📊 การทำความสะอาดข้อมูล
- ✅ จัดการข้อมูลที่หายไป (Missing data handling)
- ✅ ลบข้อมูลซ้ำ (Duplicate removal)
- ✅ ปรับมาตรฐานรูปแบบ (Format standardization)
- ✅ ตรวจจับค่าผิดปกติ (Outlier detection)
- ✅ ตรวจสอบความสอดคล้อง (Consistency validation)

### 🔄 การแปลงข้อมูล
- ✅ สร้างฟีเจอร์ใหม่ (Feature engineering)
- ✅ แปลงประเภทข้อมูล (Data type conversion)
- ✅ ปรับมาตรฐานข้อมูล (Data normalization)
- ✅ เข้ารหัสข้อมูลหมวดหมู่ (Categorical encoding)
- ✅ แยกและรวมคอลัมน์ (Column split/merge)

### 📈 การตรวจสอบคุณภาพ
- ✅ ประเมินความครบถ้วน (Completeness assessment)
- ✅ ตรวจสอบความไม่ซ้ำ (Uniqueness validation)
- ✅ วัดความสอดคล้อง (Consistency measurement)
- ✅ ตรวจสอบความถูกต้อง (Accuracy validation)
- ✅ คำนวณคะแนนคุณภาพ (Quality scoring)

### 📊 การรายงาน
- ✅ รายงาน HTML แบบโต้ตอบ (Interactive HTML reports)
- ✅ ส่งออก JSON และ TXT (JSON and TXT export)
- ✅ สรุปผู้บริหาร (Executive summary)
- ✅ รายงานทางเทคนิค (Technical reports)
- ✅ เนื้อหาภาษาไทย (Thai language content)

## 🚀 การติดตั้งและใช้งาน (Installation & Usage)

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. การใช้งานแบบโต้ตอบ (Interactive Mode)
```bash
python main_thai.py -i
```

### 3. การใช้งานแบบ Command Line
```bash
# ประมวลผลไฟล์เดียว
python main_thai.py --input data/raw/sample.csv --output data/processed/cleaned.csv

# ใช้ไฟล์การตั้งค่า
python main_thai.py --input data.csv --output cleaned.csv --config config/config.yaml --verbose
```

### 4. การสาธิต (Demo)
```bash
python demo.py
```

## 📁 โครงสร้างโครงการ (Project Structure)

```
Data Cleansing/
├── main_thai.py           # หน้าต่างหลักภาษาไทย
├── main.py               # หน้าต่างหลักภาษาอังกฤษ
├── demo.py               # การสาธิตระบบ
├── requirements.txt      # Dependencies
├── config/               # ไฟล์การตั้งค่า
│   ├── config.yaml       # การตั้งค่าหลัก
│   └── schema.json       # โครงสร้างข้อมูล
├── modules/              # โมดูลหลัก
│   ├── data_loader.py    # โหลดข้อมูล
│   ├── data_cleaner.py   # ทำความสะอาด
│   ├── data_transformer.py # แปลงข้อมูล
│   ├── data_validator.py # ตรวจสอบคุณภาพ
│   ├── reporter.py       # สร้างรายงาน
│   └── utils.py          # เครื่องมือช่วย
├── data/                 # ข้อมูล
│   ├── raw/              # ข้อมูลต้นฉบับ
│   └── processed/        # ข้อมูลที่ประมวลผลแล้ว
├── templates/            # เทมเพลต
├── tests/                # การทดสอบ
└── logs/                 # บันทึก
```

## 🎯 วิธีการใช้งาน (How to Use)

### 1. โหมดโต้ตอบ (Interactive Mode)
เมื่อรันโปรแกรมจะแสดงเมนูภาษาไทย 8 หัวข้อหลัก:

1. **📂 นำเข้าข้อมูล** - รองรับ CSV, Excel, JSON, Database
2. **👁️ ดูข้อมูล** - แสดงตัวอย่างและสถิติข้อมูล
3. **🧹 กระบวนการทำความสะอาด** - ทำความสะอาดข้อมูลแบบครบวงจร
4. **📊 ตรวจสอบและรายงานผล** - ประเมินคุณภาพและสร้างรายงาน
5. **💾 ส่งออกข้อมูล** - บันทึกข้อมูลในรูปแบบต่างๆ
6. **⚙️ ตั้งค่า** - ปรับแต่งพารามิเตอร์
7. **📋 เทมเพลต** - ใช้เทมเพลตที่กำหนดไว้
8. **📈 รายงาน** - ดูรายงานที่สร้างแล้ว

### 2. การใช้งานใน Python Script
```python
from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.data_validator import DataValidator

# โหลดข้อมูล
loader = DataLoader()
data = loader.load_data('data.csv')

# ทำความสะอาด
cleaner = DataCleaner()
result = cleaner.clean_data(data)
cleaned_data = result['cleaned_data']

# ตรวจสอบคุณภาพ
validator = DataValidator()
quality_report = validator.validate_data(cleaned_data)
print(f"Quality Score: {quality_report['quality_score']}")
```

## 📊 รูปแบบข้อมูลที่รองรับ (Supported Data Formats)

### Input Formats
- **CSV** - ไฟล์ CSV มาตรฐาน
- **Excel** - ไฟล์ .xlsx และ .xls
- **JSON** - ไฟล์ JSON หลายรูปแบบ
- **Database** - SQLite, MySQL, PostgreSQL

### Output Formats
- **CSV** - พร้อม encoding UTF-8
- **Excel** - พร้อมการจัดรูปแบบ
- **JSON** - รูปแบบ records หรือ table
- **Parquet** - สำหรับข้อมูลขนาดใหญ่

## 🔧 การตั้งค่า (Configuration)

### ไฟล์ config.yaml
```yaml
processing:
  max_rows_in_memory: 1000000
  chunk_size: 10000
  
missing_data:
  default_strategy: "drop"
  numeric_fill_value: 0
  
validation:
  quality_thresholds:
    completeness: 0.95
    uniqueness: 0.98
```

### การตั้งค่าภาษาไทย
- รองรับการตรวจสอบเลขบัตรประชาชน
- ตรวจสอบเบอร์โทรศัพท์ไทย
- รหัสไปรษณีย์ไทย
- การจัดการอักขระพิเศษภาษาไทย

## 🧪 การทดสอบ (Testing)

```bash
# รันการทดสอบทั้งหมด
python -m pytest tests/ -v

# ทดสอบโมดูลเฉพาะ
python -m pytest tests/test_data_cleaner.py -v

# รันการทดสอบด้วย coverage
python -m pytest tests/ --cov=modules
```

## 📈 ตัวอย่างผลลัพธ์ (Sample Results)

### คะแนนคุณภาพ (Quality Scores)
- **ความครบถ้วน (Completeness)**: 95.2%
- **ความไม่ซ้ำ (Uniqueness)**: 98.7%
- **ความสอดคล้อง (Consistency)**: 94.8%
- **ความถูกต้อง (Accuracy)**: 96.5%
- **คะแนนรวม (Overall Score)**: 96.3%

### การปรับปรุงข้อมูล
- ลบแถวซ้ำ: 15 แถว
- เติมข้อมูลที่หายไป: 23 เซลล์
- ปรับมาตรฐานรูปแบบ: 156 เซลล์
- ตรวจจับค่าผิดปกติ: 4 ค่า

## 🤝 การมีส่วนร่วม (Contributing)

1. Fork โปรเจกต์
2. สร้าง feature branch
3. Commit การเปลี่ยนแปลง
4. Push ไปยัง branch
5. สร้าง Pull Request

## 📝 License

โปรเจกต์นี้ได้รับอนุญาตภายใต้ MIT License

## 🆘 การสนับสนุน (Support)

หากพบปัญหาหรือต้องการความช่วยเหลือ:
- ดู Documentation ในโฟลเดอร์ `docs/`
- ตรวจสอบ Issues ที่มีอยู่
- สร้าง Issue ใหม่พร้อมรายละเอียด

## 🔄 Version History

- **v1.0.0** - เวอร์ชันเริ่มต้น
  - ระบบทำความสะอาดข้อมูลครบถ้วน
  - ส่วนติดต่อผู้ใช้ภาษาไทย
  - การรายงานแบบ HTML
  - รองรับไฟล์หลายรูปแบบ

## 🚀 แผนการพัฒนา (Roadmap)

- [ ] ส่วนติดต่อผู้ใช้แบบ GUI
- [ ] การประมวลผลแบบ batch
- [ ] การเชื่อมต่อ API ภายนอก
- [ ] รายงานแบบ PDF
- [ ] ระบบ scheduling อัตโนมัติ
- [ ] Machine Learning สำหรับการตรวจจับ anomaly

---

**พัฒนาด้วย ❤️ สำหรับชุมชนผู้พัฒนาไทย**

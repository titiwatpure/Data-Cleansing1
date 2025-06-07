# Data Cleansing Pipeline Logs
# บันทึกการทำงานของระบบทำความสะอาดข้อมูล

## โครงสร้างไฟล์ Log

ระบบจะสร้างไฟล์ log อัตโนมัติเมื่อเริ่มใช้งาน

### ตัวอย่างไฟล์ log:
- `data_cleansing_YYYYMMDD.log` - log รายวัน
- `error_YYYYMMDD.log` - log ข้อผิดพลาด
- `performance_YYYYMMDD.log` - log ประสิทธิภาพ

### รูปแบบ log message:
```
[TIMESTAMP] [LEVEL] [MODULE] - MESSAGE
```

### ระดับ log:
- DEBUG: ข้อมูลสำหรับการแก้ไขข้อบกพร่อง
- INFO: ข้อมูลทั่วไปการทำงาน
- WARNING: คำเตือน
- ERROR: ข้อผิดพลาด
- CRITICAL: ข้อผิดพลาดร้ายแรง

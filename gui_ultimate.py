import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import pandas as pd
import os
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')


class DataCleaningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧹 ระบบทำความสะอาดข้อมูลแบบครบครัน 🇹🇭")
        self.root.geometry("1400x900")
        self.data = None
        self.original_data = None
        self.cleaned_data = None
        self.cleaning_history = []
        self.undo_stack = []
        self.redo_stack = []

        # ตัวแปร UI
        self.info_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.tree = None

        # การตั้งค่าเริ่มต้น
        self.settings = {
            'auto_clean': False,
            'fill_method': 'mean',  # mean, median, mode, zero, forward, backward, interpolate, remove
            'duplicate_method': 'first',  # first, last, all
            'outlier_method': 'zscore',  # zscore, iqr, isolation_forest, none
            'outlier_threshold': 3.0,
            'date_format': '%Y-%m-%d',
            'encoding': 'utf-8-sig',
            'decimal_places': 2,
            'show_preview': True,
            'backup_original': True,
            'working_mode': 'standard',  # beginner, standard, expert
            'max_display_rows': 1000,
            'auto_detect_types': True,
            'standardize_text': True,
            'remove_whitespace': True,
            'clean_email': True,
            'clean_phone': True,
            'validate_dates': True,
            'custom_fill_values': {},
            'column_specific_settings': {}
        }

        # โหลดการตั้งค่าที่บันทึกไว้
        self.load_settings()

        # ตั้งค่าสไตล์
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.create_menu()
        self.create_widgets()

    def create_widgets(self):
        # หัวข้อหลัก
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)

        title_label = ttk.Label(title_frame, text="🧹 ระบบทำความสะอาดข้อมูลแบบครบครัน 🇹🇭",
                                font=("Arial", 18, "bold"))
        title_label.pack()

        # แถบข้อมูลสรุป
        self.create_info_panel()

        # ปุ่มควบคุม
        self.create_control_panel()

        # พื้นที่แสดงข้อมูล
        self.create_data_panel()

        # แถบสถานะ
        self.create_status_bar()

    def create_info_panel(self):
        info_frame = ttk.LabelFrame(self.root, text="📋 ข้อมูลสรุป")
        info_frame.pack(fill='x', padx=10, pady=5)

        self.info_var = tk.StringVar()
        self.info_var.set(
            "🔍 ยังไม่ได้โหลดข้อมูล - กรุณาเลือกไฟล์เพื่อเริ่มต้น")

        info_label = ttk.Label(info_frame, textvariable=self.info_var,
                               font=("Arial", 11), foreground="blue")
        info_label.pack(pady=8)

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(
            self.root, text="🛠️ เครื่องมือทำความสะอาด")
        control_frame.pack(fill='x', padx=10, pady=5)

        # แถวที่ 1 - ไฟล์และการทำความสะอาดหลัก
        row1 = ttk.Frame(control_frame)
        row1.pack(fill='x', padx=10, pady=5)

        ttk.Button(row1, text="📂 เปิดไฟล์", command=self.open_file,
                   width=15).pack(side='left', padx=3)
        ttk.Button(row1, text="💾 บันทึก", command=self.save_file,
                   width=15).pack(side='left', padx=3)
        ttk.Button(row1, text="🧹 ทำความสะอาดทั้งหมด",
                   command=self.clean_all, width=20).pack(side='left', padx=3)
        ttk.Button(row1, text="📊 สถิติข้อมูล", command=self.show_stats,
                   width=15).pack(side='left', padx=3)

        # แถวที่ 2 - เครื่องมือเฉพาะ
        row2 = ttk.Frame(control_frame)
        row2.pack(fill='x', padx=10, pady=5)

        ttk.Button(row2, text="🗑️ ลบข้อมูลซ้ำ",
                   command=self.remove_duplicates, width=15).pack(side='left', padx=3)
        ttk.Button(row2, text="🔧 เติมค่าว่าง", command=self.fill_missing,
                   width=15).pack(side='left', padx=3)
        ttk.Button(row2, text="🧼 ลบแถวว่าง", command=self.remove_empty_rows,
                   width=15).pack(side='left', padx=3)
        ttk.Button(row2, text="📋 เปรียบเทียบ", command=self.compare_data,
                   width=15).pack(side='left', padx=3)

        # แถวที่ 3 - เครื่องมือเพิ่มเติม
        row3 = ttk.Frame(control_frame)
        row3.pack(fill='x', padx=10, pady=5)

        ttk.Button(row3, text="🔄 รีเซ็ต", command=self.reset_data,
                   width=15).pack(side='left', padx=3)
        ttk.Button(row3, text="📄 ส่งออกรายงาน",
                   command=self.export_report, width=15).pack(side='left', padx=3)
        ttk.Button(row3, text="🎯 ตรวจสอบคุณภาพ",
                   command=self.validate_data, width=15).pack(side='left', padx=3)
        ttk.Button(row3, text="❌ ปิดโปรแกรม", command=self.root.quit,
                   width=15).pack(side='left', padx=3)

    def create_data_panel(self):
        # พื้นที่แสดงข้อมูล
        data_frame = ttk.LabelFrame(self.root, text="📄 ข้อมูล")
        data_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Treeview สำหรับแสดงข้อมูล
        tree_frame = ttk.Frame(data_frame)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.tree = ttk.Treeview(tree_frame)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(
            tree_frame, orient="vertical", command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(
            tree_frame, orient="horizontal", command=self.tree.xview)

        self.tree.configure(yscrollcommand=v_scrollbar.set,
                            xscrollcommand=h_scrollbar.set)

        # จัดวางใน Grid
        self.tree.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def create_status_bar(self):
        # สถานะ
        self.status_var = tk.StringVar()
        self.status_var.set("✅ พร้อมใช้งาน - เลือกไฟล์เพื่อเริ่มต้น")

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill='x', padx=10, pady=5)

        status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                 relief='sunken', font=("Arial", 10))
        status_label.pack(fill='x')

    def open_file(self):
        file_path = filedialog.askopenfilename(
            title="เลือกไฟล์ข้อมูล",
            filetypes=[
                ("Excel files", "*.xlsx *.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                # โหลดข้อมูล
                if file_path.endswith('.csv'):
                    # ลองหลายการเข้ารหัส
                    encodings = ['utf-8', 'utf-8-sig', 'cp874', 'iso-8859-1']
                    data_loaded = False

                    for encoding in encodings:
                        try:
                            self.data = pd.read_csv(
                                file_path, encoding=encoding)
                            data_loaded = True
                            break
                        except UnicodeDecodeError:
                            continue

                    if not data_loaded:
                        self.data = pd.read_csv(
                            file_path, encoding='utf-8', encoding_errors='ignore')
                else:
                    self.data = pd.read_excel(file_path)

                if self.data is not None:
                    self.original_data = self.data.copy()
                    self.cleaning_history = []
                    self.update_display()
                    self.update_info_panel()

                    filename = os.path.basename(file_path)
                    self.status_var.set(f"✅ โหลดไฟล์สำเร็จ: {filename}")
                else:
                    raise ValueError("ไม่สามารถโหลดข้อมูลได้")

            except Exception as e:
                messagebox.showerror(
                    "ข้อผิดพลาด", f"ไม่สามารถเปิดไฟล์ได้:\n{str(e)}")

    def update_info_panel(self):
        if self.data is not None:
            rows, cols = self.data.shape
            missing = self.data.isnull().sum().sum()
            duplicates = self.data.duplicated().sum()
            memory_mb = self.data.memory_usage(deep=True).sum() / 1024 / 1024

            info_text = (f"📊 แถว: {rows:,} | คอลัมน์: {cols:,} | "
                         f"ค่าว่าง: {missing:,} | ซ้ำ: {duplicates:,} | "
                         f"ขนาด: {memory_mb:.1f} MB")

            self.info_var.set(info_text)

    def save_file(self):
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้บันทึก")
            return

        file_path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data.to_csv(file_path, index=False,
                                     encoding='utf-8-sig')
                else:
                    self.data.to_excel(file_path, index=False)
                filename = os.path.basename(file_path)
                self.status_var.set(f"✅ บันทึกไฟล์สำเร็จ: {filename}")
                messagebox.showinfo("สำเร็จ", f"บันทึกไฟล์สำเร็จ: {filename}")

            except Exception as e:
                messagebox.showerror(
                    "ข้อผิดพลาด", f"ไม่สามารถบันทึกไฟล์ได้:\n{str(e)}")

    def clean_all(self):
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ทำความสะอาด")
            return

        # ยืนยันการทำความสะอาด
        result = messagebox.askyesno("ยืนยัน",
                                     "คุณต้องการทำความสะอาดข้อมูลทั้งหมดหรือไม่?\n"
                                     "การกระทำนี้จะไม่สามารถยกเลิกได้")
        if not result:
            return

        try:
            original_rows = len(self.data)

            # บันทึกประวัติ
            self.cleaning_history.append(f"เริ่มต้น: {original_rows:,} แถว")

            # 1. ลบแถวที่ว่างทั้งหมด
            before_empty = len(self.data)
            self.data = self.data.dropna(how='all')
            empty_removed = before_empty - len(self.data)
            if empty_removed > 0:
                self.cleaning_history.append(
                    f"ลบแถวว่าง: {empty_removed:,} แถว")

            # 2. ลบข้อมูลซ้ำ
            before_dup = len(self.data)
            self.data = self.data.drop_duplicates()
            dup_removed = before_dup - len(self.data)
            if dup_removed > 0:
                self.cleaning_history.append(
                    f"ลบข้อมูลซ้ำ: {dup_removed:,} แถว")

            # 3. เติมค่าที่หายไป
            missing_before = self.data.isnull().sum().sum()

            for col in self.data.columns:
                if self.data[col].dtype in ['int64', 'float64']:
                    # ใช้ค่าเฉลี่ยสำหรับตัวเลข
                    mean_val = self.data[col].mean()
                    if pd.notna(mean_val):
                        self.data[col] = self.data[col].fillna(mean_val)
                else:
                    # ใช้ค่าที่พบบ่อยที่สุดสำหรับข้อความ
                    try:
                        mode_val = self.data[col].mode()
                        if len(mode_val) > 0 and pd.notna(mode_val.iloc[0]):
                            self.data[col] = self.data[col].fillna(
                                mode_val.iloc[0])
                        else:
                            self.data[col] = self.data[col].fillna('ไม่ระบุ')
                    except (ValueError, IndexError, TypeError):
                        self.data[col] = self.data[col].fillna('ไม่ระบุ')
                        self.data[col] = self.data[col].fillna('ไม่ระบุ')

            missing_after = self.data.isnull().sum().sum()
            filled = missing_before - missing_after
            if filled > 0:
                self.cleaning_history.append(f"เติมค่าว่าง: {filled:,} ค่า")

            # 4. ปรับปรุงรูปแบบข้อมูล
            self.standardize_data_types()

            cleaned_rows = len(self.data)
            total_removed = original_rows - cleaned_rows

            self.update_display()
            self.update_info_panel()

            # สร้างรายงาน
            report = f"""🎉 ทำความสะอาดข้อมูลเสร็จสิ้น!

📊 สรุปผลลัพธ์:
• แถวเดิม: {original_rows:,}
• แถวหลังทำความสะอาด: {cleaned_rows:,}
• แถวที่ลบทั้งหมด: {total_removed:,}

📝 รายละเอียด:
{chr(10).join('• ' + item for item in self.cleaning_history)}

✅ ข้อมูลพร้อมใช้งานแล้ว!"""

            self.status_var.set(
                f"✅ ทำความสะอาดเสร็จ: เหลือ {cleaned_rows:,} แถว")
            messagebox.showinfo("สำเร็จ", report)

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการทำความสะอาด:\n{str(e)}")

    def standardize_data_types(self):
        """ปรับปรุงรูปแบบข้อมูลให้เป็นมาตรฐาน"""
        if self.data is None:
            return

        for col in self.data.columns:
            # ลองแปลงวันที่
            if self.data[col].dtype == 'object':
                try:
                    # ลองแปลงเป็นวันที่
                    pd.to_datetime(self.data[col], errors='raise')
                    self.data[col] = pd.to_datetime(
                        self.data[col], errors='coerce')
                    self.cleaning_history.append(
                        f"แปลงคอลัมน์ '{col}' เป็นวันที่")
                except (ValueError, TypeError):
                    # ลองแปลงเป็นตัวเลข
                    try:
                        numeric_data = pd.to_numeric(
                            self.data[col], errors='raise')
                        self.data[col] = numeric_data
                        self.cleaning_history.append(
                            f"แปลงคอลัมน์ '{col}' เป็นตัวเลข")
                    except (ValueError, TypeError):
                        pass

    def remove_duplicates(self):
        """ลบข้อมูลซ้ำ"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
            return

        try:
            self.save_to_undo_stack()
            original_count = len(self.data)

            # ลบข้อมูลซ้ำตามการตั้งค่า
            method = self.settings.get('duplicate_method', 'first')
            if method == 'first':
                self.data = self.data.drop_duplicates(keep='first')
            elif method == 'last':
                self.data = self.data.drop_duplicates(keep='last')
            else:  # 'all'
                self.data = self.data.drop_duplicates(keep=False)

            removed_count = original_count - len(self.data)

            self.update_display()
            self.update_info_panel()

            if removed_count > 0:
                self.cleaning_history.append(
                    f"ลบข้อมูลซ้ำ: {removed_count:,} แถว")
                self.status_var.set(
                    f"✅ ลบข้อมูลซ้ำสำเร็จ: {removed_count:,} แถว")
                messagebox.showinfo(
                    "สำเร็จ", f"ลบข้อมูลซ้ำ {removed_count:,} แถว")
            else:
                messagebox.showinfo("แจ้งเตือน", "ไม่พบข้อมูลซ้ำ")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถลบข้อมูลซ้ำได้:\n{str(e)}")

    def fill_missing(self):
        """เติมค่าว่าง"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
            return

        try:
            self.save_to_undo_stack()
            missing_before = self.data.isnull().sum().sum()

            if missing_before == 0:
                messagebox.showinfo("แจ้งเตือน", "ไม่มีค่าว่างในข้อมูล")
                return

            # เติมค่าว่างตามการตั้งค่า
            method = self.settings.get('fill_method', 'mean')

            for col in self.data.columns:
                if self.data[col].isnull().any():
                    if self.data[col].dtype in ['int64', 'float64']:
                        if method == 'mean':
                            self.data[col].fillna(
                                self.data[col].mean(), inplace=True)
                        elif method == 'median':
                            self.data[col].fillna(
                                self.data[col].median(), inplace=True)
                        elif method == 'zero':
                            self.data[col].fillna(0, inplace=True)
                    else:
                        if method == 'mode':
                            mode_val = self.data[col].mode()
                            if len(mode_val) > 0:
                                self.data[col].fillna(
                                    mode_val[0], inplace=True)
                        else:
                            self.data[col].fillna('ไม่ระบุ', inplace=True)

            missing_after = self.data.isnull().sum().sum()
            filled_count = missing_before - missing_after

            self.update_display()
            self.update_info_panel()

            if filled_count > 0:
                self.cleaning_history.append(
                    f"เติมค่าว่าง: {filled_count:,} ค่า")
                self.status_var.set(
                    f"✅ เติมค่าว่างสำเร็จ: {filled_count:,} ค่า")
                messagebox.showinfo(
                    "สำเร็จ", f"เติมค่าว่าง {filled_count:,} ค่า")
            else:
                messagebox.showinfo("แจ้งเตือน", "ไม่สามารถเติมค่าว่างได้")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถเติมค่าว่างได้:\n{str(e)}")

    def remove_empty_rows(self):
        """ลบแถวที่ว่างทั้งหมด"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
            return

        try:
            self.save_to_undo_stack()
            original_count = len(self.data)
            self.data = self.data.dropna(how='all')
            removed_count = original_count - len(self.data)

            self.update_display()
            self.update_info_panel()

            if removed_count > 0:
                self.cleaning_history.append(
                    f"ลบแถวว่าง: {removed_count:,} แถว")
                self.status_var.set(
                    f"✅ ลบแถวว่างสำเร็จ: {removed_count:,} แถว")
                messagebox.showinfo(
                    "สำเร็จ", f"ลบแถวว่าง {removed_count:,} แถว")
            else:
                messagebox.showinfo("แจ้งเตือน", "ไม่พบแถวที่ว่างทั้งหมด")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถลบแถวว่างได้:\n{str(e)}")

    def compare_data(self):
        """เปรียบเทียบข้อมูลก่อนและหลังทำความสะอาด"""
        if self.data is None or self.original_data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้เปรียบเทียบ")
            return

        compare_window = tk.Toplevel(self.root)
        compare_window.title("📋 เปรียบเทียบข้อมูล")
        compare_window.geometry("800x600")

        notebook = ttk.Notebook(compare_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # แท็บสรุป
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📊 สรุป")

        summary_text = tk.Text(
            summary_frame, wrap=tk.WORD, font=("Consolas", 10))
        summary_scrollbar = ttk.Scrollbar(
            summary_frame, orient="vertical", command=summary_text.yview)
        summary_text.configure(yscrollcommand=summary_scrollbar.set)

        # สร้างข้อความเปรียบเทียบ
        comparison = self.generate_comparison_report()
        summary_text.insert('1.0', comparison)
        summary_text.config(state='disabled')

        summary_text.pack(side='left', fill='both', expand=True)
        summary_scrollbar.pack(side='right', fill='y')

    def generate_comparison_report(self):
        """สร้างรายงานเปรียบเทียบ"""
        original = self.original_data
        current = self.data

        # Check if data exists
        if original is None or current is None:
            return "❌ ไม่สามารถสร้างรายงานเปรียบเทียบได้: ไม่มีข้อมูล"

        report = "📋 รายงานเปรียบเทียบข้อมูล\n"
        report += "=" * 50 + "\n\n"

        # เปรียบเทียบพื้นฐาน
        report += "📊 ข้อมูลพื้นฐาน:\n"
        report += f"จำนวนแถว: {len(original):,} → {len(current):,} ({len(current) - len(original):+,})\n"
        report += f"จำนวนคอลัมน์: {len(original.columns):,} → {len(current.columns):,}\n"

        # เปรียบเทียบค่าว่าง
        original_missing = original.isnull().sum().sum()
        current_missing = current.isnull().sum().sum()
        report += f"ค่าว่างทั้งหมด: {original_missing:,} → {current_missing:,} ({current_missing - original_missing:+,})\n"

        # เปรียบเทียบข้อมูลซ้ำ
        original_duplicates = original.duplicated().sum()
        current_duplicates = current.duplicated().sum()
        report += f"ข้อมูลซ้ำ: {original_duplicates:,} → {current_duplicates:,} ({current_duplicates - original_duplicates:+,})\n\n"

        # ประวัติการทำความสะอาด
        if self.cleaning_history:
            report += "📝 ประวัติการทำความสะอาด:\n"
            for i, action in enumerate(self.cleaning_history, 1):
                report += f"{i}. {action}\n"
        else:
            report += "📝 ยังไม่มีการทำความสะอาด\n"

        return report

    def export_report(self):
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ส่งออกรายงาน")
            return

        try:
            # สร้างรายงาน
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            report_data = {
                'สรุปข้อมูล': {
                    'จำนวนแถว': len(self.data),
                    'จำนวนคอลัมน์': len(self.data.columns),
                    'ค่าว่างทั้งหมด': self.data.isnull().sum().sum(),
                    'ข้อมูลซ้ำ': self.data.duplicated().sum(),
                    'วันที่สร้างรายงาน': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            # สถิติแต่ละคอลัมน์
            column_stats = {}
            for col in self.data.columns:
                stats = {
                    'ประเภทข้อมูล': str(self.data[col].dtype),
                    'ค่าที่ไม่ว่าง': self.data[col].count(),
                    'ค่าว่าง': self.data[col].isnull().sum(),
                    'เปอร์เซ็นต์ว่าง': round(self.data[col].isnull().sum() / len(self.data) * 100, 2)
                }

                if self.data[col].dtype in ['int64', 'float64']:
                    stats.update({
                        'ค่าเฉลี่ย': round(self.data[col].mean(), 2) if pd.notna(self.data[col].mean()) else None,
                        'ค่าสูงสุด': self.data[col].max(),
                        'ค่าต่ำสุด': self.data[col].min(),
                        'ค่าเบี่ยงเบนมาตรฐาน': round(self.data[col].std(), 2) if pd.notna(self.data[col].std()) else None
                    })
                else:
                    stats.update({
                        'ค่าที่ไม่ซ้ำ': self.data[col].nunique()
                    })

                column_stats[col] = stats

            # เลือกตำแหน่งบันทึก
            file_path = filedialog.asksaveasfilename(
                title="ส่งออกรายงาน",
                defaultextension=".xlsx",
                filetypes=[
                    ("Excel files", "*.xlsx"),
                    ("CSV files", "*.csv")],
                initialfile=f"รายงานข้อมูล_{timestamp}"
            )

            if file_path:
                if file_path.endswith('.xlsx'):
                    # ส่งออกเป็น Excel
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        # ข้อมูลหลัก
                        self.data.to_excel(
                            writer, sheet_name='ข้อมูลหลัก', index=False)

                        # สรุปข้อมูล
                        summary_df = pd.DataFrame(
                            [report_data['สรุปข้อมูล']]).T
                        summary_df.columns = ['ค่า']
                        summary_df.to_excel(writer, sheet_name='สรุปข้อมูล')

                        # สถิติคอลัมน์
                        stats_df = pd.DataFrame(column_stats).T
                        stats_df.to_excel(writer, sheet_name='สถิติคอลัมน์')

                        # ประวัติการทำความสะอาด
                        if self.cleaning_history:
                            history_df = pd.DataFrame(
                                self.cleaning_history, columns=['การกระทำ'])
                            history_df.to_excel(
                                writer, sheet_name='ประวัติ', index=False)
                else:
                    # ส่งออกเป็น CSV
                    self.data.to_csv(file_path, index=False,
                                     encoding='utf-8-sig')

                filename = os.path.basename(file_path)
                self.status_var.set(f"✅ ส่งออกรายงานสำเร็จ: {filename}")
                messagebox.showinfo(
                    "สำเร็จ", f"ส่งออกรายงานสำเร็จ!\n\nไฟล์: {filename}")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถส่งออกรายงานได้:\n{str(e)}")

    def validate_data(self):
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ตรวจสอบ")
            return

        validate_window = tk.Toplevel(self.root)
        validate_window.title("🎯 ตรวจสอบคุณภาพข้อมูล")
        validate_window.geometry("700x500")

        text_frame = ttk.Frame(validate_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(
            text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        # ตรวจสอบคุณภาพข้อมูล
        quality_score = 100
        issues = []

        # ตรวจสอบค่าว่าง
        missing_pct = (self.data.isnull().sum().sum() /
                       (len(self.data) * len(self.data.columns))) * 100
        if missing_pct > 10:
            quality_score -= 20
            issues.append(f"❌ ค่าว่างมากเกินไป ({missing_pct:.1f}%)")
        elif missing_pct > 5:
            quality_score -= 10
            issues.append(f"⚠️ ค่าว่างปานกลาง ({missing_pct:.1f}%)")
        else:
            issues.append(f"✅ ค่าว่างในระดับที่ยอมรับได้ ({missing_pct:.1f}%)")

        # ตรวจสอบข้อมูลซ้ำ
        dup_pct = (self.data.duplicated().sum() / len(self.data)) * 100
        if dup_pct > 5:
            quality_score -= 15
            issues.append(f"❌ ข้อมูลซ้ำมากเกินไป ({dup_pct:.1f}%)")
        elif dup_pct > 0:
            quality_score -= 5
            issues.append(f"⚠️ มีข้อมูลซ้ำ ({dup_pct:.1f}%)")
        else:
            issues.append("✅ ไม่มีข้อมูลซ้ำ")

        # ตรวจสอบความสอดคล้องของประเภทข้อมูล
        type_issues = 0
        for col in self.data.columns:
            # ตรวจสอบว่าควรเป็นตัวเลขหรือไม่
            if self.data[col].dtype == 'object':
                try:
                    pd.to_numeric(self.data[col], errors='raise')
                    type_issues += 1
                    issues.append(f"⚠️ คอลัมน์ '{col}' ควรเป็นตัวเลข")
                except (ValueError, TypeError, AttributeError):
                    pass

        if type_issues > 0:
            quality_score -= (type_issues * 5)

        # กำหนดเกรด
        if quality_score >= 90:
            grade = "A"
            grade_color = "🟢"
        elif quality_score >= 80:
            grade = "B"
            grade_color = "🟡"
        elif quality_score >= 70:
            grade = "C"
            grade_color = "🟠"
        else:
            grade = "D"
            grade_color = "🔴"

        report = f"""🎯 รายงานการตรวจสอบคุณภาพข้อมูล
{'='*50}

{grade_color} คะแนนคุณภาพ: {quality_score}/100 (เกรด {grade})

📊 สรุปผล:
• จำนวนแถว: {len(self.data):,}
• จำนวนคอลัมน์: {len(self.data.columns):,}
• ค่าว่างทั้งหมด: {self.data.isnull().sum().sum():,} ({missing_pct:.1f}%)
• ข้อมูลซ้ำ: {self.data.duplicated().sum():,} ({dup_pct:.1f}%)

🔍 รายการปัญหาที่พบ:
"""

        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue}\n"

        if quality_score < 90:
            report += "\n💡 คำแนะนำ:\n"
            if missing_pct > 5:
                report += "• ใช้ฟังก์ชัน 'เติมค่าว่าง' หรือ 'ลบแถวว่าง'\n"
            if dup_pct > 0:
                report += "• ใช้ฟังก์ชัน 'ลบข้อมูลซ้ำ'\n"
            if type_issues > 0:
                report += "• ตรวจสอบและแปลงประเภทข้อมูลให้ถูกต้อง\n"
        else:
            report += "\n🎉 ยินดีด้วย! ข้อมูลมีคุณภาพดีมาก\n"

        text_widget.insert(tk.END, report)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Button(validate_window, text="ปิด",
                   command=validate_window.destroy).pack(pady=10)

    def reset_data(self):
        if self.original_data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลต้นฉบับให้รีเซ็ต")
            return

        result = messagebox.askyesno(
            "ยืนยัน", "คุณต้องการรีเซ็ตข้อมูลกลับเป็นต้นฉบับหรือไม่?")
        if result:
            self.data = self.original_data.copy()
            self.cleaning_history = []
            self.update_display()
            self.update_info_panel()
            self.status_var.set("✅ รีเซ็ตข้อมูลเรียบร้อย")

    def show_stats(self):
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้แสดงสถิติ")
            return

        try:
            stats_window = tk.Toplevel(self.root)
            stats_window.title("📊 สถิติข้อมูล")
            stats_window.geometry("700x600")

            # สร้าง Text widget พร้อม Scrollbar
            text_frame = ttk.Frame(stats_window)
            text_frame.pack(fill='both', expand=True, padx=10, pady=10)

            text_widget = tk.Text(
                text_frame, wrap=tk.WORD, font=("Consolas", 10))
            scrollbar = ttk.Scrollbar(
                text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(
                yscrollcommand=scrollbar.set)            # สถิติทั่วไป
            memory_mb = self.data.memory_usage(deep=True).sum() / 1024 / 1024
            stats_info = f"""📊 สถิติข้อมูลทั่วไป
{'='*60}
📋 ข้อมูลพื้นฐาน:
• จำนวนแถว: {len(self.data):,}
• จำนวนคอลัมน์: {len(self.data.columns):,}
• ขนาดข้อมูล: {memory_mb:.2f} MB
• ค่าว่างทั้งหมด: {self.data.isnull().sum().sum():,}
• ข้อมูลซ้ำ: {self.data.duplicated().sum():,}

📋 รายละเอียดแต่ละคอลัมน์:
{'='*60}
"""

            for col in self.data.columns:
                stats_info += f"\n🔹 {col}\n"
                stats_info += f"   ├─ ประเภท: {self.data[col].dtype}\n"
                stats_info += f"   ├─ ค่าที่ไม่ว่าง: {self.data[col].count():,}\n"
                stats_info += f"   ├─ ค่าว่าง: {self.data[col].isnull().sum():,}\n"
                stats_info += f"   └─ เปอร์เซ็นต์ว่าง: {(self.data[col].isnull().sum() / len(self.data) * 100):.1f}%\n"

                if self.data[col].dtype in ['int64', 'float64']:
                    stats_info += "   📈 สถิติตัวเลข:\n"
                    stats_info += f"      ├─ ค่าเฉลี่ย: {self.data[col].mean():.2f}\n"
                    stats_info += f"      ├─ ค่ากลาง: {self.data[col].median():.2f}\n"
                    stats_info += f"      ├─ ค่าสูงสุด: {self.data[col].max()}\n"
                    stats_info += f"      ├─ ค่าต่ำสุด: {self.data[col].min()}\n"
                    stats_info += f"      └─ ส่วนเบี่ยงเบน: {self.data[col].std():.2f}\n"
                else:
                    unique_count = self.data[col].nunique()
                    stats_info += "   📝 สถิติข้อความ:\n"
                    stats_info += f"      ├─ ค่าที่ไม่ซ้ำ: {unique_count:,}\n"

                    if unique_count <= 10 and unique_count > 0:
                        top_values = self.data[col].value_counts().head(5)
                        stats_info += "      └─ ค่าที่พบบ่อย:\n"
                        for val, count in top_values.items():
                            percentage = (count / len(self.data)) * 100
                            stats_info += f"         • {val}: {count:,} ({percentage:.1f}%)\n"

            text_widget.insert(tk.END, stats_info)
            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # ปุ่มปิด
            ttk.Button(stats_window, text="ปิด",
                       command=stats_window.destroy).pack(pady=5)

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถแสดงสถิติได้:\n{str(e)}")

    def update_display(self):
        """อัปเดตการแสดงข้อมูลในตาราง"""
        if self.data is None or self.tree is None:
            return

        try:
            # ลบข้อมูลเดิม
            for item in self.tree.get_children():
                self.tree.delete(item)

            # ตั้งค่าคอลัมน์
            columns = list(self.data.columns)
            self.tree["columns"] = columns
            self.tree["show"] = "headings"

            # ตั้งค่า heading และ width
            for col in columns:
                self.tree.heading(col, text=col)
                if col.lower() in ['id', 'no', 'index']:
                    self.tree.column(
                        col, width=60, minwidth=60, anchor='center')
                else:
                    self.tree.column(col, width=120, minwidth=80)

            # จำกัดการแสดงผลตามการตั้งค่า
            max_rows = self.settings.get('max_display_rows', 1000)
            display_data = self.data.head(max_rows)

            # เพิ่มข้อมูล
            for index, row in display_data.iterrows():
                values = [str(val) if pd.notna(val) else '' for val in row]
                self.tree.insert("", "end", values=values)

            # แสดงข้อความแจ้งเตือนถ้าข้อมูลมากเกินไป
            if len(self.data) > max_rows:
                notice_values = ['แสดงเพียง {} แถวแรก'.format(
                    max_rows)] + [''] * (len(self.data.columns) - 1)
                self.tree.insert("", "end", values=notice_values)

        except Exception as e:
            print(f"ข้อผิดพลาดในการอัปเดตการแสดงผล: {e}")

    def create_menu(self):
        """สร้างเมนูบาร์"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # เมนูไฟล์
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="📁 ไฟล์", menu=file_menu)
        file_menu.add_command(label="📂 เปิดไฟล์",
                              command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(
            label="💾 บันทึก", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="💾 บันทึกเป็น...",
                              command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="📊 นำเข้าตัวอย่าง",
                              command=self.load_sample_data)
        file_menu.add_separator()
        file_menu.add_command(
            label="❌ ออก", command=self.root.quit, accelerator="Ctrl+Q")

        # เมนูแก้ไข
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="✏️ แก้ไข", menu=edit_menu)
        edit_menu.add_command(
            label="🔄 เลิกทำ", command=self.undo_action, accelerator="Ctrl+Z")
        edit_menu.add_command(
            label="🔄 ทำซ้ำ", command=self.redo_action, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="🔄 รีเซ็ตข้อมูล", command=self.reset_data)

        # เมนูเครื่องมือ
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🛠️ เครื่องมือ", menu=tools_menu)

        # เมนูย่อยการทำความสะอาด
        clean_submenu = tk.Menu(tools_menu, tearoff=0)
        tools_menu.add_cascade(label="🧹 ทำความสะอาด", menu=clean_submenu)
        clean_submenu.add_command(
            label="🧹 ทำความสะอาดทั้งหมด", command=self.clean_all)
        clean_submenu.add_command(
            label="🗑️ ลบข้อมูลซ้ำ", command=self.remove_duplicates)
        clean_submenu.add_command(
            label="🔧 เติมค่าว่าง", command=self.fill_missing)
        clean_submenu.add_command(
            label="🧼 ลบแถวว่าง", command=self.remove_empty_rows)
        clean_submenu.add_command(
            label="🎯 ตรวจจับค่าผิดปกติ", command=self.detect_outliers)
        clean_submenu.add_command(
            label="📏 มาตรฐานข้อมูล", command=self.standardize_data)

        tools_menu.add_separator()
        tools_menu.add_command(label="📊 สถิติข้อมูล", command=self.show_stats)
        tools_menu.add_command(label="📋 เปรียบเทียบข้อมูล",
                               command=self.compare_data)
        tools_menu.add_command(label="🎯 ตรวจสอบคุณภาพ",
                               command=self.validate_data)
        tools_menu.add_command(label="📄 ส่งออกรายงาน",
                               command=self.export_report)

        # เมนูตั้งค่า
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="⚙️ ตั้งค่า", menu=settings_menu)
        settings_menu.add_command(
            label="🎛️ การตั้งค่าทั่วไป", command=self.open_settings)
        settings_menu.add_command(
            label="🧹 ตั้งค่าการทำความสะอาด", command=self.open_cleaning_settings)
        settings_menu.add_command(
            label="📊 ตั้งค่าการแสดงผล", command=self.open_display_settings)
        settings_menu.add_command(
            label="🎮 เลือกโหมดการทำงาน", command=self.select_working_mode)
        settings_menu.add_separator()
        settings_menu.add_command(
            label="💾 บันทึกการตั้งค่า", command=self.save_settings)
        settings_menu.add_command(
            label="🔄 รีเซ็ตการตั้งค่า", command=self.reset_settings)

        # เมนูช่วยเหลือ
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="❓ ช่วยเหลือ", menu=help_menu)
        help_menu.add_command(label="📖 คู่มือการใช้งาน",
                              command=self.show_help)
        help_menu.add_command(label="💡 เคล็ดลับ", command=self.show_tips)
        help_menu.add_command(label="🔍 ตัวอย่างการใช้งาน",
                              command=self.show_examples)
        help_menu.add_separator()
        help_menu.add_command(label="ℹ️ เกี่ยวกับโปรแกรม",
                              command=self.show_about)

        # คีย์บอร์ดชอร์ตคัต
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-z>', lambda e: self.undo_action())
        self.root.bind('<Control-y>', lambda e: self.redo_action())
        self.root.bind('<Control-q>', lambda e: self.root.quit())

    def load_settings(self):
        """โหลดการตั้งค่าจากไฟล์"""
        try:
            if os.path.exists('settings.json'):
                with open('settings.json', 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    self.settings.update(saved_settings)
        except Exception as e:
            print(f"ไม่สามารถโหลดการตั้งค่าได้: {e}")

    def save_settings(self):
        """บันทึกการตั้งค่าลงไฟล์"""
        try:
            with open('settings.json', 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("สำเร็จ", "บันทึกการตั้งค่าเรียบร้อย")
        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถบันทึกการตั้งค่าได้: {e}")

    def reset_settings(self):
        """รีเซ็ตการตั้งค่าเป็นค่าเริ่มต้น"""
        if messagebox.askyesno("ยืนยัน", "ต้องการรีเซ็ตการตั้งค่าทั้งหมดหรือไม่?"):
            self.settings = {
                'auto_clean': False,
                'fill_method': 'mean',
                'duplicate_method': 'first',
                'outlier_method': 'zscore',
                'outlier_threshold': 3.0,
                'date_format': '%Y-%m-%d',
                'encoding': 'utf-8-sig',
                'decimal_places': 2,
                'show_preview': True,
                'backup_original': True,
                'working_mode': 'standard'
            }
            messagebox.showinfo("สำเร็จ", "รีเซ็ตการตั้งค่าเรียบร้อย")

    def open_settings(self):
        """เปิดหน้าต่างการตั้งค่าทั่วไป"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ การตั้งค่าทั่วไป")
        settings_window.geometry("500x400")
        settings_window.resizable(False, False)

        # ทำให้หน้าต่างอยู่ตรงกลาง
        settings_window.transient(self.root)
        settings_window.grab_set()

        notebook = ttk.Notebook(settings_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # แท็บทั่วไป
        general_frame = ttk.Frame(notebook)
        notebook.add(general_frame, text="ทั่วไป")

        # การสำรองข้อมูล
        ttk.Label(general_frame, text="🛡️ การสำรองข้อมูล", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))
        backup_var = tk.BooleanVar(value=self.settings['backup_original'])
        ttk.Checkbutton(general_frame, text="สำรองข้อมูลต้นฉบับอัตโนมัติ",
                        variable=backup_var).pack(anchor='w', padx=20)

        # การแสดงตัวอย่าง
        ttk.Label(general_frame, text="👀 การแสดงผล", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        preview_var = tk.BooleanVar(value=self.settings['show_preview'])
        ttk.Checkbutton(general_frame, text="แสดงตัวอย่างก่อนดำเนินการ",
                        variable=preview_var).pack(anchor='w', padx=20)

        # การเข้ารหัส
        ttk.Label(general_frame, text="📝 การเข้ารหัสไฟล์", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        encoding_var = tk.StringVar(value=self.settings['encoding'])
        encoding_frame = ttk.Frame(general_frame)
        encoding_frame.pack(anchor='w', padx=20)
        ttk.Label(encoding_frame, text="Encoding:").pack(side='left')
        encoding_combo = ttk.Combobox(encoding_frame, textvariable=encoding_var,
                                      values=['utf-8-sig', 'utf-8',
                                              'cp874', 'iso-8859-1'],
                                      width=15, state='readonly')
        encoding_combo.pack(side='left', padx=(10, 0))

        # ทศนิยม
        ttk.Label(general_frame, text="🔢 จำนวนทศนิยม", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        decimal_var = tk.IntVar(value=self.settings['decimal_places'])
        decimal_frame = ttk.Frame(general_frame)
        decimal_frame.pack(anchor='w', padx=20)
        ttk.Label(decimal_frame, text="ตำแหน่งทศนิยม:").pack(side='left')
        ttk.Spinbox(decimal_frame, from_=0, to=10, textvariable=decimal_var,
                    width=10).pack(side='left', padx=(10, 0))

        # ปุ่มบันทึก
        button_frame = ttk.Frame(general_frame)
        button_frame.pack(fill='x', pady=(30, 10))

        def save_general_settings():
            self.settings['backup_original'] = backup_var.get()
            self.settings['show_preview'] = preview_var.get()
            self.settings['encoding'] = encoding_var.get()
            self.settings['decimal_places'] = decimal_var.get()
            messagebox.showinfo("สำเร็จ", "บันทึกการตั้งค่าเรียบร้อย")
            settings_window.destroy()

        ttk.Button(button_frame, text="💾 บันทึก",
                   command=save_general_settings).pack(side='right', padx=5)
        ttk.Button(button_frame, text="❌ ยกเลิก",
                   command=settings_window.destroy).pack(side='right')

    def open_cleaning_settings(self):
        """เปิดหน้าต่างตั้งค่าการทำความสะอาด"""
        cleaning_window = tk.Toplevel(self.root)
        cleaning_window.title("🧹 ตั้งค่าการทำความสะอาด")
        cleaning_window.geometry("600x500")
        cleaning_window.resizable(False, False)
        cleaning_window.transient(self.root)
        cleaning_window.grab_set()

        notebook = ttk.Notebook(cleaning_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # แท็บการเติมค่าว่าง
        fill_frame = ttk.Frame(notebook)
        notebook.add(fill_frame, text="เติมค่าว่าง")

        ttk.Label(fill_frame, text="🔧 วิธีการเติมค่าว่าง", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))
        fill_var = tk.StringVar(value=self.settings['fill_method'])

        fill_options = [
            ('mean', 'ค่าเฉลี่ย (สำหรับตัวเลข)'),
            ('median', 'ค่ากลาง (สำหรับตัวเลข)'),
            ('mode', 'ค่าที่พบบ่อยที่สุด'),
            ('zero', 'ใส่ค่า 0'),
            ('remove', 'ลบแถวที่มีค่าว่าง')
        ]

        for value, text in fill_options:
            ttk.Radiobutton(fill_frame, text=text, variable=fill_var, value=value).pack(
                anchor='w', padx=20, pady=2)

        # แท็บข้อมูลซ้ำ
        dup_frame = ttk.Frame(notebook)
        notebook.add(dup_frame, text="ข้อมูลซ้ำ")

        ttk.Label(dup_frame, text="🗑️ วิธีการจัดการข้อมูลซ้ำ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))
        dup_var = tk.StringVar(value=self.settings['duplicate_method'])

        dup_options = [
            ('first', 'เก็บข้อมูลแรก ลบข้อมูลซ้ำที่เหลือ'),
            ('last', 'เก็บข้อมูลสุดท้าย ลบข้อมูลซ้ำก่อนหน้า'),
            ('all', 'ลบข้อมูลซ้ำทั้งหมด')
        ]

        for value, text in dup_options:
            ttk.Radiobutton(dup_frame, text=text, variable=dup_var, value=value).pack(
                anchor='w', padx=20, pady=2)

        # แท็บค่าผิดปกติ
        outlier_frame = ttk.Frame(notebook)
        notebook.add(outlier_frame, text="ค่าผิดปกติ")

        ttk.Label(outlier_frame, text="🎯 วิธีการตรวจจับค่าผิดปกติ",
                  font=("Arial", 12, "bold")).pack(anchor='w', pady=(10, 5))
        outlier_var = tk.StringVar(value=self.settings['outlier_method'])

        outlier_options = [
            ('zscore', 'Z-Score (ระยะห่างจากค่าเฉลี่ย)'),
            ('iqr', 'IQR (Inter Quartile Range)'),
            ('none', 'ไม่ตรวจจับ')
        ]

        for value, text in outlier_options:
            ttk.Radiobutton(outlier_frame, text=text, variable=outlier_var, value=value).pack(
                anchor='w', padx=20, pady=2)

        # ค่า threshold
        ttk.Label(outlier_frame, text="📏 ค่าความไว (Threshold)", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 5))
        threshold_var = tk.DoubleVar(value=self.settings['outlier_threshold'])
        threshold_frame = ttk.Frame(outlier_frame)
        threshold_frame.pack(anchor='w', padx=20)
        ttk.Label(threshold_frame, text="ค่า Threshold:").pack(side='left')
        ttk.Spinbox(threshold_frame, from_=1.0, to=5.0, increment=0.1,
                    textvariable=threshold_var, width=10).pack(side='left', padx=(10, 0))

        # ปุ่มบันทึก
        button_frame = ttk.Frame(cleaning_window)
        button_frame.pack(fill='x', padx=10, pady=10)

        def save_cleaning_settings():
            self.settings['fill_method'] = fill_var.get()
            self.settings['duplicate_method'] = dup_var.get()
            self.settings['outlier_method'] = outlier_var.get()
            self.settings['outlier_threshold'] = threshold_var.get()
            messagebox.showinfo(
                "สำเร็จ", "บันทึกการตั้งค่าการทำความสะอาดเรียบร้อย")
            cleaning_window.destroy()

        ttk.Button(button_frame, text="💾 บันทึก",
                   command=save_cleaning_settings).pack(side='right', padx=5)
        ttk.Button(button_frame, text="❌ ยกเลิก",
                   command=cleaning_window.destroy).pack(side='right')

    def select_working_mode(self):
        """เลือกโหมดการทำงาน"""
        mode_window = tk.Toplevel(self.root)
        mode_window.title("🎮 เลือกโหมดการทำงาน")
        mode_window.geometry("500x400")
        mode_window.resizable(False, False)
        mode_window.transient(self.root)
        mode_window.grab_set()

        ttk.Label(mode_window, text="🎮 เลือกโหมดการทำงาน",
                  font=("Arial", 16, "bold")).pack(pady=20)

        mode_var = tk.StringVar(value=self.settings['working_mode'])

        # โหมดผู้เริ่มต้น
        beginner_frame = ttk.LabelFrame(
            mode_window, text="🌱 โหมดผู้เริ่มต้น", padding=10)
        beginner_frame.pack(fill='x', padx=20, pady=10)
        ttk.Radiobutton(beginner_frame, text="เหมาะสำหรับผู้ใช้ใหม่ มีคำแนะนำและตัวเลือกพื้นฐาน",
                        variable=mode_var, value='beginner').pack(anchor='w')

        # โหมดมาตรฐาน
        standard_frame = ttk.LabelFrame(
            mode_window, text="⚙️ โหมดมาตรฐาน", padding=10)
        standard_frame.pack(fill='x', padx=20, pady=10)
        ttk.Radiobutton(standard_frame, text="โหมดปกติ เหมาะสำหรับผู้ใช้ทั่วไป",
                        variable=mode_var, value='standard').pack(anchor='w')

        # โหมดผู้เชี่ยวชาญ
        expert_frame = ttk.LabelFrame(
            mode_window, text="🚀 โหมดผู้เชี่ยวชาญ", padding=10)
        expert_frame.pack(fill='x', padx=20, pady=10)
        ttk.Radiobutton(expert_frame, text="ตัวเลือกครบถ้วน เหมาะสำหรับผู้เชี่ยวชาญ",
                        variable=mode_var, value='expert').pack(anchor='w')

        def apply_mode():
            self.settings['working_mode'] = mode_var.get()
            messagebox.showinfo("สำเร็จ", f"เปลี่ยนเป็นโหมด: {mode_var.get()}")
            mode_window.destroy()

        button_frame = ttk.Frame(mode_window)
        button_frame.pack(fill='x', padx=20, pady=20)
        ttk.Button(button_frame, text="✅ ยืนยัน",
                   command=apply_mode).pack(side='right', padx=5)
        ttk.Button(button_frame, text="❌ ยกเลิก",
                   command=mode_window.destroy).pack(side='right')

    def open_display_settings(self):
        """เปิดหน้าต่างตั้งค่าการแสดงผล"""
        display_window = tk.Toplevel(self.root)
        display_window.title("📊 ตั้งค่าการแสดงผล")
        display_window.geometry("450x350")
        display_window.resizable(False, False)
        display_window.transient(self.root)
        display_window.grab_set()

        # ตั้งค่าแถวที่จะแสดง
        ttk.Label(display_window, text="📄 จำนวนแถวที่แสดง", font=(
            "Arial", 12, "bold")).pack(anchor='w', padx=20, pady=(20, 5))
        rows_var = tk.IntVar(value=1000)
        rows_frame = ttk.Frame(display_window)
        rows_frame.pack(anchor='w', padx=20)
        ttk.Label(rows_frame, text="แถว:").pack(side='left')
        ttk.Spinbox(rows_frame, from_=100, to=10000, increment=100,
                    textvariable=rows_var, width=10).pack(side='left', padx=(10, 0))

        # รูปแบบตัวเลข
        ttk.Label(display_window, text="🔢 รูปแบบตัวเลข", font=(
            "Arial", 12, "bold")).pack(anchor='w', padx=20, pady=(20, 5))
        number_format_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_window, text="แสดงเครื่องหมายจุลภาค (1,000)",
                        variable=number_format_var).pack(anchor='w', padx=30)

        # แสดงสถิติ
        ttk.Label(display_window, text="📈 การแสดงสถิติ", font=(
            "Arial", 12, "bold")).pack(anchor='w', padx=20, pady=(20, 5))
        auto_stats_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_window, text="แสดงสถิติอัตโนมัติเมื่อโหลดข้อมูล",
                        variable=auto_stats_var).pack(anchor='w', padx=30)

        # ปุ่มบันทึก
        button_frame = ttk.Frame(display_window)
        button_frame.pack(fill='x', padx=20, pady=30)

        def save_display_settings():
            messagebox.showinfo("สำเร็จ", "บันทึกการตั้งค่าการแสดงผลเรียบร้อย")
            display_window.destroy()

        ttk.Button(button_frame, text="💾 บันทึก",
                   command=save_display_settings).pack(side='right', padx=5)
        ttk.Button(button_frame, text="❌ ยกเลิก",
                   command=display_window.destroy).pack(side='right')

    def load_sample_data(self):
        """โหลดข้อมูลตัวอย่างสำหรับทดสอบ"""
        try:
            # สร้างข้อมูลตัวอย่าง
            sample_data = {
                'ลำดับ': range(1, 101),
                'ชื่อ': [f'ผู้ใช้ {i}' for i in range(1, 101)],
                'อายุ': np.random.randint(18, 65, 100),
                'เงินเดือน': np.random.randint(15000, 100000, 100),
                'แผนก': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 100),
                'วันเริ่มงาน': pd.date_range('2020-01-01', periods=100, freq='D')
            }

            # เพิ่มข้อมูลที่มีปัญหา
            sample_df = pd.DataFrame(sample_data)

            # เพิ่มค่าว่าง
            sample_df.loc[10:15, 'อายุ'] = np.nan
            sample_df.loc[20:25, 'เงินเดือน'] = np.nan

            # เพิ่มข้อมูลซ้ำ
            sample_df = pd.concat(
                [sample_df, sample_df.iloc[0:5]], ignore_index=True)

            # เพิ่มค่าผิดปกติ
            sample_df.loc[50, 'เงินเดือน'] = 1000000  # outlier
            # outlier            self.data = sample_df
            sample_df.loc[51, 'อายุ'] = 150
            if self.data is not None:
                self.original_data = self.data.copy()
                self.cleaning_history = []

                self.update_display()
                self.update_info_panel()
                self.status_var.set("✅ โหลดข้อมูลตัวอย่างสำเร็จ")
            else:
                raise ValueError("ไม่สามารถสร้างข้อมูลตัวอย่างได้")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถสร้างข้อมูลตัวอย่างได้:\n{str(e)}")

    def save_as_file(self):
        """บันทึกไฟล์เป็นชื่อใหม่"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้บันทึก")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = filedialog.asksaveasfilename(
            title="บันทึกไฟล์เป็น",
            initialdir=os.getcwd(),
            initialfile=f"ข้อมูลสะอาด_{timestamp}",
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data.to_csv(file_path, index=False,
                                     encoding='utf-8-sig')
                else:
                    self.data.to_excel(file_path, index=False)

                filename = os.path.basename(file_path)
                self.status_var.set(f"✅ บันทึกไฟล์สำเร็จ: {filename}")
                messagebox.showinfo("สำเร็จ", f"บันทึกไฟล์สำเร็จ: {filename}")

            except Exception as e:
                messagebox.showerror(
                    "ข้อผิดพลาด", f"ไม่สามารถบันทึกไฟล์ได้:\n{str(e)}")

    def undo_action(self):
        """ยกเลิกการกระทำล่าสุด"""
        if not self.undo_stack:
            messagebox.showinfo("แจ้งเตือน", "ไม่มีการกระทำที่จะยกเลิก")
            return

        # เก็บสถานะปัจจุบันลง redo stack
        if self.data is not None:
            self.redo_stack.append(self.data.copy())

        # คืนสถานะก่อนหน้า
        self.data = self.undo_stack.pop()
        self.update_display()
        self.update_info_panel()
        self.status_var.set("↶ ยกเลิกการกระทำล่าสุด")

    def redo_action(self):
        """ทำซ้ำการกระทำที่ยกเลิกไป"""
        if not self.redo_stack:
            messagebox.showinfo("แจ้งเตือน", "ไม่มีการกระทำที่จะทำซ้ำ")
            return

        # เก็บสถานะปัจจุบันลง undo stack
        if self.data is not None:
            self.undo_stack.append(self.data.copy())

        # คืนสถานะที่ยกเลิกไป
        self.data = self.redo_stack.pop()
        self.update_display()
        self.update_info_panel()
        self.status_var.set("↷ ทำซ้ำการกระทำ")

    def detect_outliers(self):
        """ตรวจจับและจัดการค่าผิดปกติ"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
            return

        try:
            # เปิดหน้าต่างตัวเลือก
            outlier_window = tk.Toplevel(self.root)
            outlier_window.title("🎯 ตรวจจับค่าผิดปกติ")
            outlier_window.geometry("500x400")

            # ตัวเลือกวิธีการ
            method_frame = ttk.LabelFrame(
                outlier_window, text="วิธีการตรวจจับ")
            method_frame.pack(fill='x', padx=10, pady=10)

            method_var = tk.StringVar(
                value=self.settings.get('outlier_method', 'zscore'))
            ttk.Radiobutton(method_frame, text="Z-Score",
                            variable=method_var, value='zscore').pack(anchor='w')
            ttk.Radiobutton(method_frame, text="IQR",
                            variable=method_var, value='iqr').pack(anchor='w')

            # เกณฑ์
            threshold_frame = ttk.Frame(method_frame)
            threshold_frame.pack(fill='x', pady=5)
            ttk.Label(threshold_frame, text="เกณฑ์:").pack(side='left')
            threshold_var = tk.DoubleVar(
                value=self.settings.get('outlier_threshold', 3.0))
            ttk.Spinbox(threshold_frame, from_=1.0, to=5.0, increment=0.1,
                        textvariable=threshold_var, width=10).pack(side='left', padx=5)

            # การจัดการ            action_frame = ttk.LabelFrame(outlier_window, text="การจัดการ")
            action_frame.pack(fill='x', padx=10, pady=10)

            action_var = tk.StringVar(value='remove')
            ttk.Radiobutton(action_frame, text="ลบค่าผิดปกติ",
                            variable=action_var, value='remove').pack(anchor='w')
            ttk.Radiobutton(action_frame, text="แสดงรายการเท่านั้น",
                            variable=action_var, value='show').pack(anchor='w')

            def apply_outlier_detection():
                try:
                    if self.data is None:
                        messagebox.showwarning(
                            "คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
                        return

                    outliers_found = 0
                    outlier_details = []

                    # Get numeric columns safely
                    try:
                        numeric_columns = self.data.select_dtypes(
                            include=[np.number]).columns
                    except (AttributeError, TypeError):
                        messagebox.showerror(
                            "ข้อผิดพลาด", "ไม่สามารถเข้าถึงคอลัมน์ตัวเลขได้")
                        return

                    for col in numeric_columns:
                        try:
                            if method_var.get() == 'zscore':
                                col_data = self.data[col]
                                if col_data is None or col_data.empty:
                                    continue

                                mean_val = col_data.mean()
                                std_val = col_data.std()

                                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                                    continue

                                z_scores = np.abs(
                                    (col_data - mean_val) / std_val)
                                outliers = z_scores > threshold_var.get()
                            # IQR                                col_data = self.data[col]
                            else:
                                if col_data is None or col_data.empty:
                                    continue

                                q1 = col_data.quantile(0.25)
                                q3 = col_data.quantile(0.75)

                                if pd.isna(q1) or pd.isna(q3):
                                    continue

                                iqr = q3 - q1
                                lower_bound = q1 - 1.5 * iqr
                                upper_bound = q3 + 1.5 * iqr
                                outliers = (col_data < lower_bound) | (
                                    col_data > upper_bound)
                        except Exception as col_error:
                            print(
                                f"Error processing column {col}: {str(col_error)}")
                            continue

                        try:
                            outlier_count = outliers.sum()
                            if outlier_count > 0:
                                outliers_found += outlier_count
                                outlier_details.append(
                                    f"คอลัมน์ {col}: {outlier_count} ค่า")
                        except Exception as count_error:
                            print(
                                f"Error counting outliers for column {col}: {str(count_error)}")
                            continue

                    if outliers_found > 0:
                        if action_var.get() == 'remove':
                            self.save_to_undo_stack()
                            # ลบค่าผิดปกติ (ต้องปรับปรุงให้ละเอียดกว่านี้)
                            result_text = f"พบค่าผิดปกติ {outliers_found} ค่า:\n" + "\n".join(
                                outlier_details)
                        else:
                            result_text = f"พบค่าผิดปกติ {outliers_found} ค่า:\n" + "\n".join(
                                outlier_details)

                        messagebox.showinfo("ผลการตรวจจับ", result_text)
                    else:
                        messagebox.showinfo("ผลการตรวจจับ", "ไม่พบค่าผิดปกติ")

                    outlier_window.destroy()

                except Exception as e:
                    messagebox.showerror(
                        "ข้อผิดพลาด", f"เกิดข้อผิดพลาด:\n{str(e)}")

            # ปุ่ม
            button_frame = ttk.Frame(outlier_window)
            button_frame.pack(fill='x', padx=10, pady=10)
            ttk.Button(button_frame, text="ดำเนินการ",
                       command=apply_outlier_detection).pack(side='left', padx=5)
            ttk.Button(button_frame, text="ยกเลิก", command=outlier_window.destroy).pack(
                side='right', padx=5)

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถตรวจจับค่าผิดปกติได้:\n{str(e)}")

    def standardize_data(self):
        """ปรับมาตรฐานข้อมูล"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ประมวลผล")
            return

        try:
            self.save_to_undo_stack()
            changes = []

            # ปรับข้อความ
            for col in self.data.select_dtypes(include=['object']).columns:
                if self.settings.get('standardize_text', True):
                    # ลบช่องว่างที่ไม่จำเป็น
                    if self.settings.get('remove_whitespace', True):
                        self.data[col] = self.data[col].astype(str).str.strip()
                        changes.append(f"ลบช่องว่างในคอลัมน์ {col}")

            # ปรับตัวเลข
            decimal_places = self.settings.get('decimal_places', 2)
            for col in self.data.select_dtypes(include=['float64']).columns:
                self.data[col] = self.data[col].round(decimal_places)
                changes.append(
                    f"ปรับทศนิยมคอลัมน์ {col} เป็น {decimal_places} ตำแหน่ง")

            self.update_display()
            self.update_info_panel()

            if changes:
                self.cleaning_history.extend(changes)
                result_text = "ปรับมาตรฐานข้อมูลสำเร็จ:\n" + \
                    "\n".join(f"• {change}" for change in changes)
                messagebox.showinfo("สำเร็จ", result_text)
                self.status_var.set("✅ ปรับมาตรฐานข้อมูลเรียบร้อย")
            else:
                messagebox.showinfo("แจ้งเตือน", "ไม่มีการเปลี่ยนแปลง")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถปรับมาตรฐานข้อมูลได้:\n{str(e)}")

    def open_advanced_cleaning_options(self):
        """เปิดหน้าต่างตัวเลือกการทำความสะอาดขั้นสูง"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดข้อมูลก่อน")
            return

        options_window = tk.Toplevel(self.root)
        options_window.title("🛠️ ตัวเลือกการทำความสะอาดขั้นสูง")
        options_window.geometry("800x700")
        options_window.resizable(True, True)

        # Notebook สำหรับแท็บต่างๆ
        notebook = ttk.Notebook(options_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # แท็บ 1: การจัดการค่าว่าง
        missing_frame = ttk.Frame(notebook)
        notebook.add(missing_frame, text="🔧 ค่าว่าง")
        self.create_missing_data_options(missing_frame)

        # แท็บ 2: การจัดการข้อมูลซ้ำ
        duplicate_frame = ttk.Frame(notebook)
        notebook.add(duplicate_frame, text="🗑️ ข้อมูลซ้ำ")
        self.create_duplicate_options(duplicate_frame)

        # แท็บ 3: การตรวจจับค่าผิดปกติ
        outlier_frame = ttk.Frame(notebook)
        notebook.add(outlier_frame, text="🎯 ค่าผิดปกติ")
        self.create_outlier_options(outlier_frame)

        # แท็บ 4: การมาตรฐานข้อมูล
        standardize_frame = ttk.Frame(notebook)
        notebook.add(standardize_frame, text="📏 มาตรฐาน")
        self.create_standardization_options(standardize_frame)

        # แท็บ 5: การตั้งค่าตามคอลัมน์
        column_frame = ttk.Frame(notebook)
        notebook.add(column_frame, text="📊 ตามคอลัมน์")
        self.create_column_specific_options(column_frame)

        # ปุ่มดำเนินการ
        button_frame = ttk.Frame(options_window)
        button_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(button_frame, text="🔍 แสดงตัวอย่าง",
                   command=lambda: self.preview_cleaning_changes()).pack(side='left', padx=5)
        ttk.Button(button_frame, text="✅ ดำเนินการ",
                   command=lambda: self.apply_advanced_cleaning(options_window)).pack(side='left', padx=5)
        ttk.Button(button_frame, text="💾 บันทึกการตั้งค่า",
                   command=self.save_cleaning_template).pack(side='left', padx=5)
        ttk.Button(button_frame, text="❌ ยกเลิก",
                   command=options_window.destroy).pack(side='right', padx=5)

    def create_missing_data_options(self, parent):
        """สร้างตัวเลือกการจัดการค่าว่าง"""        # Create main frame with scrollable functionality
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(
            parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = scrollable_frame
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # วิธีการเติมค่าว่าง
        ttk.Label(main_frame, text="🔧 วิธีการเติมค่าว่าง", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(0, 10))

        self.fill_method_var = tk.StringVar(
            value=self.settings.get('fill_method', 'mean'))
        methods = [
            ('mean', 'ค่าเฉลี่ย (สำหรับตัวเลข)'),
            ('median', 'ค่ากลาง (สำหรับตัวเลข)'),
            ('mode', 'ค่าที่พบบ่อยที่สุด'),
            ('zero', 'ใส่ค่า 0'),
            ('forward', 'ใช้ค่าก่อนหน้า'),
            ('backward', 'ใช้ค่าถัดไป'),
            ('interpolate', 'การประมาณค่า'),
            ('custom', 'กำหนดค่าเอง'),
            ('remove', 'ลบแถวที่มีค่าว่าง')
        ]

        for value, text in methods:
            ttk.Radiobutton(main_frame, text=text, variable=self.fill_method_var,
                            value=value).pack(anchor='w', padx=20)

        # ค่าที่กำหนดเอง
        custom_frame = ttk.LabelFrame(
            main_frame, text="ค่าที่กำหนดเองตามคอลัมน์")
        custom_frame.pack(fill='x', pady=(20, 0))

        self.custom_values = {}
        if self.data is not None:
            # แสดงเพียง 10 คอลัมน์แรก
            for i, col in enumerate(self.data.columns[:10]):
                row_frame = ttk.Frame(custom_frame)
                row_frame.pack(fill='x', padx=5, pady=2)

                ttk.Label(row_frame, text=f"{col}:",
                          width=20).pack(side='left')
                entry = ttk.Entry(row_frame, width=30)
                entry.pack(side='left', padx=(10, 0))
                self.custom_values[col] = entry

    def create_duplicate_options(self, parent):
        """สร้างตัวเลือกการจัดการข้อมูลซ้ำ"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # วิธีการจัดการข้อมูลซ้ำ
        ttk.Label(main_frame, text="🗑️ วิธีการจัดการข้อมูลซ้ำ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(0, 10))

        self.duplicate_method_var = tk.StringVar(
            value=self.settings.get('duplicate_method', 'first'))
        dup_methods = [
            ('first', 'เก็บข้อมูลแรก ลบข้อมูลซ้ำที่เหลือ'),
            ('last', 'เก็บข้อมูลสุดท้าย ลบข้อมูลซ้ำก่อนหน้า'),
            ('all', 'ลบข้อมูลซ้ำทั้งหมด (รวมต้นฉบับ)'),
            ('mark_only', 'ทำเครื่องหมายไว้เท่านั้น ไม่ลบ')
        ]

        for value, text in dup_methods:
            ttk.Radiobutton(main_frame, text=text, variable=self.duplicate_method_var,
                            value=value).pack(anchor='w', padx=20)

        # เลือกคอลัมน์สำหรับตรวจสอบข้อมูลซ้ำ
        ttk.Label(main_frame, text="📊 เลือกคอลัมน์สำหรับตรวจสอบ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 10))

        self.duplicate_columns = {}
        if self.data is not None:
            columns_frame = ttk.Frame(main_frame)
            columns_frame.pack(fill='x')

            # สร้าง Checkbutton สำหรับแต่ละคอลัมน์
            for i, col in enumerate(self.data.columns):
                var = tk.BooleanVar(value=True)
                ttk.Checkbutton(columns_frame, text=col, variable=var).grid(
                    row=i//3, column=i % 3, sticky='w', padx=5, pady=2)
                self.duplicate_columns[col] = var

    def create_outlier_options(self, parent):
        """สร้างตัวเลือกการตรวจจับค่าผิดปกติ"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # วิธีการตรวจจับ
        ttk.Label(main_frame, text="🎯 วิธีการตรวจจับค่าผิดปกติ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(0, 10))

        self.outlier_method_var = tk.StringVar(
            value=self.settings.get('outlier_method', 'zscore'))
        outlier_methods = [
            ('zscore', 'Z-Score (ค่าเบี่ยงเบนมาตรฐาน)'),
            ('iqr', 'IQR (Interquartile Range)'),
            ('isolation_forest', 'Isolation Forest (Machine Learning)'),
            ('none', 'ไม่ตรวจจับค่าผิดปกติ')
        ]

        for value, text in outlier_methods:
            ttk.Radiobutton(main_frame, text=text, variable=self.outlier_method_var,
                            value=value).pack(anchor='w', padx=20)

        # การตั้งค่าเกณฑ์
        threshold_frame = ttk.LabelFrame(main_frame, text="เกณฑ์การตรวจจับ")
        threshold_frame.pack(fill='x', pady=(20, 0))

        # Z-Score threshold
        zscore_frame = ttk.Frame(threshold_frame)
        zscore_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(zscore_frame, text="Z-Score Threshold:").pack(side='left')
        self.zscore_threshold = tk.DoubleVar(
            value=self.settings.get('outlier_threshold', 3.0))
        ttk.Spinbox(zscore_frame, from_=1.0, to=5.0, increment=0.1,
                    textvariable=self.zscore_threshold, width=10).pack(side='left', padx=(10, 0))

        # การจัดการค่าผิดปกติ
        ttk.Label(main_frame, text="🛠️ วิธีการจัดการค่าผิดปกติ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 10))

        self.outlier_action_var = tk.StringVar(value='remove')
        outlier_actions = [
            ('remove', 'ลบข้อมูลผิดปกติ'),
            ('cap', 'จำกัดค่าสูงสุด/ต่ำสุด'),
            ('transform', 'แปลงข้อมูล (Log transformation)'),
            ('mark_only', 'ทำเครื่องหมายไว้เท่านั้น')
        ]

        for value, text in outlier_actions:
            ttk.Radiobutton(main_frame, text=text, variable=self.outlier_action_var,
                            value=value).pack(anchor='w', padx=20)

    def create_standardization_options(self, parent):
        """สร้างตัวเลือกการมาตรฐานข้อมูล"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # การจัดการข้อความ
        ttk.Label(main_frame, text="📝 การจัดการข้อความ", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(0, 10))

        self.text_options = {}
        text_settings = [
            ('remove_whitespace', 'ลบช่องว่างที่ไม่จำเป็น'),
            ('standardize_case', 'ปรับตัวพิมพ์ให้เหมือนกัน'),
            ('remove_special_chars', 'ลบอักขระพิเศษ'),
            ('standardize_encoding', 'ปรับการเข้ารหัสให้เหมือนกัน'),
            ('clean_email', 'ทำความสะอาดอีเมล'),
            ('clean_phone', 'ทำความสะอาดเบอร์โทรศัพท์'),
            ('clean_url', 'ทำความสะอาด URL')
        ]

        for key, text in text_settings:
            var = tk.BooleanVar(value=self.settings.get(key, True))
            ttk.Checkbutton(main_frame, text=text, variable=var).pack(
                anchor='w', padx=20)
            self.text_options[key] = var

        # การจัดการวันที่
        ttk.Label(main_frame, text="📅 การจัดการวันที่", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 10))

        date_frame = ttk.Frame(main_frame)
        date_frame.pack(fill='x', padx=20)

        ttk.Label(date_frame, text="รูปแบบวันที่มาตรฐาน:").pack(side='left')
        self.date_format_var = tk.StringVar(
            value=self.settings.get('date_format', '%Y-%m-%d'))
        date_formats = ['%Y-%m-%d', '%d/%m/%Y',
                        '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']
        ttk.Combobox(date_frame, textvariable=self.date_format_var, values=date_formats,
                     width=20).pack(side='left', padx=(10, 0))

        # การจัดการตัวเลข
        ttk.Label(main_frame, text="🔢 การจัดการตัวเลข", font=(
            "Arial", 12, "bold")).pack(anchor='w', pady=(20, 10))

        number_frame = ttk.Frame(main_frame)
        number_frame.pack(fill='x', padx=20)

        ttk.Label(number_frame, text="จำนวนทศนิยม:").pack(side='left')
        self.decimal_places_var = tk.IntVar(
            value=self.settings.get('decimal_places', 2))
        ttk.Spinbox(number_frame, from_=0, to=10, textvariable=self.decimal_places_var,
                    width=5).pack(side='left', padx=(10, 0))

    def create_column_specific_options(self, parent):
        """สร้างตัวเลือกการตั้งค่าตามคอลัมน์"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        if self.data is None:
            ttk.Label(main_frame, text="ไม่มีข้อมูลให้แสดง").pack()
            return

        # ตาราง Canvas สำหรับ scroll
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(
            main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # หัวตาราง
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill='x', padx=5, pady=5)

        ttk.Label(header_frame, text="คอลัมน์", width=15, font=(
            "Arial", 10, "bold")).grid(row=0, column=0, padx=5)
        ttk.Label(header_frame, text="ประเภทข้อมูล", width=15, font=(
            "Arial", 10, "bold")).grid(row=0, column=1, padx=5)
        ttk.Label(header_frame, text="วิธีเติมค่าว่าง", width=15, font=(
            "Arial", 10, "bold")).grid(row=0, column=2, padx=5)
        ttk.Label(header_frame, text="ค่าเริ่มต้น", width=15, font=(
            "Arial", 10, "bold")).grid(row=0, column=3, padx=5)
        ttk.Label(header_frame, text="การตรวจสอบ", width=15, font=(
            "Arial", 10, "bold")).grid(row=0, column=4, padx=5)

        # ข้อมูลแต่ละคอลัมน์
        self.column_settings = {}
        for i, col in enumerate(self.data.columns):
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill='x', padx=5, pady=2)

            # ชื่อคอลัมน์
            ttk.Label(row_frame, text=col[:12] + "..." if len(col) > 12 else col,
                      width=15).grid(row=0, column=0, padx=5, sticky='w')

            # ประเภทข้อมูล
            dtype = str(self.data[col].dtype)
            ttk.Label(row_frame, text=dtype, width=15).grid(
                row=0, column=1, padx=5, sticky='w')

            # วิธีเติมค่าว่าง
            fill_var = tk.StringVar(value='auto')
            fill_combo = ttk.Combobox(row_frame, textvariable=fill_var,
                                      values=['auto', 'mean', 'median',
                                              'mode', 'zero', 'custom', 'remove'],
                                      width=12, state='readonly')
            fill_combo.grid(row=0, column=2, padx=5)

            # ค่าเริ่มต้น
            default_var = tk.StringVar()
            default_entry = ttk.Entry(
                row_frame, textvariable=default_var, width=15)
            default_entry.grid(row=0, column=3, padx=5)

            # การตรวจสอบ
            validate_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(row_frame, variable=validate_var).grid(
                row=0, column=4, padx=5)

            self.column_settings[col] = {
                'fill_method': fill_var,
                'default_value': default_var,
                'validate': validate_var
            }

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def preview_cleaning_changes(self):
        """แสดงตัวอย่างการเปลี่ยนแปลงก่อนทำความสะอาด"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้แสดงตัวอย่าง")
            return

        preview_window = tk.Toplevel(self.root)
        preview_window.title("🔍 ตัวอย่างการเปลี่ยนแปลง")
        preview_window.geometry("1000x600")

        notebook = ttk.Notebook(preview_window)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # แท็บข้อมูลเดิม
        original_frame = ttk.Frame(notebook)
        notebook.add(original_frame, text="📊 ข้อมูลเดิม")

        original_tree = ttk.Treeview(original_frame)
        # แสดงข้อมูลเดิม (10 แถวแรก)
        original_tree.pack(fill='both', expand=True)
        if self.data is not None:
            self.display_data_in_tree(original_tree, self.data.head(10))
        else:
            self.display_data_in_tree(original_tree, pd.DataFrame())

        # แท็บข้อมูลหลังทำความสะอาด (จำลอง)
        cleaned_frame = ttk.Frame(notebook)
        notebook.add(cleaned_frame, text="✨ หลังทำความสะอาด")

        cleaned_tree = ttk.Treeview(cleaned_frame)
        # จำลองการทำความสะอาด
        cleaned_tree.pack(fill='both', expand=True)
        preview_data = self.simulate_cleaning()
        if preview_data is not None:
            self.display_data_in_tree(cleaned_tree, preview_data.head(10))
        else:
            self.display_data_in_tree(cleaned_tree, pd.DataFrame())

        # แท็บสรุปการเปลี่ยนแปลง
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="📋 สรุป")

        summary_text = tk.Text(summary_frame, wrap=tk.WORD)
        summary_text.pack(fill='both', expand=True, padx=5, pady=5)

        # สร้างข้อความสรุป
        summary = self.generate_cleaning_summary(self.data, preview_data)
        summary_text.insert('1.0', summary)
        summary_text.config(state='disabled')

    def simulate_cleaning(self):
        """จำลองการทำความสะอาดข้อมูล"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดไฟล์ข้อมูลก่อน")
            return self.data

        try:
            # สำเนาข้อมูลสำหรับจำลอง
            preview_data = self.data.copy()

            # จำลองการเติมค่าว่าง
            fill_method = getattr(self, 'fill_method_var', None)
            if fill_method and fill_method.get() != 'remove':
                for col in preview_data.select_dtypes(include=[np.number]).columns:
                    if fill_method.get() == 'mean':
                        preview_data[col].fillna(
                            preview_data[col].mean(), inplace=True)
                    elif fill_method.get() == 'median':
                        preview_data[col].fillna(
                            preview_data[col].median(), inplace=True)
                    elif fill_method.get() == 'zero':
                        preview_data[col].fillna(0, inplace=True)

            # จำลองการลบข้อมูลซ้ำ
            duplicate_method = getattr(self, 'duplicate_method_var', None)
            if duplicate_method and duplicate_method.get() != 'mark_only':
                if duplicate_method.get() == 'first':
                    preview_data = preview_data.drop_duplicates(keep='first')
                elif duplicate_method.get() == 'last':
                    preview_data = preview_data.drop_duplicates(keep='last')
                elif duplicate_method.get() == 'all':
                    preview_data = preview_data.drop_duplicates(keep=False)

            return preview_data

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถจำลองการทำความสะอาดได้: {str(e)}")
            return self.data.copy()

    def generate_cleaning_summary(self, original, cleaned):
        """สร้างข้อความสรุปการเปลี่ยนแปลง"""
        summary = "📋 สรุปการเปลี่ยนแปลง\n"
        summary += "=" * 50 + "\n\n"

        # เปรียบเทียบจำนวนแถว
        original_rows = len(original)
        cleaned_rows = len(cleaned)
        summary += "📊 จำนวนแถว:\n"
        summary += f"   ก่อน: {original_rows:,} แถว\n"
        summary += f"   หลัง: {cleaned_rows:,} แถว\n"
        summary += f"   เปลี่ยนแปลง: {cleaned_rows - original_rows:+,} แถว\n\n"

        # เปรียบเทียบค่าว่าง
        original_missing = original.isnull().sum().sum()
        cleaned_missing = cleaned.isnull().sum().sum()
        summary += "🔧 ค่าว่าง:\n"
        summary += f"   ก่อน: {original_missing:,} ค่า\n"
        summary += f"   หลัง: {cleaned_missing:,} ค่า\n"
        summary += f"   เปลี่ยนแปลง: {cleaned_missing - original_missing:+,} ค่า\n\n"

        # เปรียบเทียบข้อมูลซ้ำ
        original_duplicates = original.duplicated().sum()
        cleaned_duplicates = cleaned.duplicated().sum()
        summary += "🗑️ ข้อมูลซ้ำ:\n"
        summary += f"   ก่อน: {original_duplicates:,} แถว\n"
        summary += f"   หลัง: {cleaned_duplicates:,} แถว\n"
        summary += f"   เปลี่ยนแปลง: {cleaned_duplicates - original_duplicates:+,} แถว\n\n"

        return summary

    def display_data_in_tree(self, tree, data):
        """แสดงข้อมูลใน Treeview"""
        # ลบข้อมูลเดิม
        for item in tree.get_children():
            tree.delete(item)

        # ตั้งค่าคอลัมน์
        columns = list(data.columns)
        tree["columns"] = columns
        tree["show"] = "headings"

        # ตั้งค่า heading
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=50)

        # เพิ่มข้อมูล
        for index, row in data.iterrows():
            values = [str(val) if pd.notna(val) else '' for val in row]
            tree.insert("", "end", values=values)

    def apply_advanced_cleaning(self, options_window):
        """ดำเนินการทำความสะอาดตามตัวเลือกที่เลือก"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ไม่มีข้อมูลให้ทำความสะอาด")
            return

        try:
            # สำรองข้อมูลเดิม
            self.save_to_undo_stack()

            original_rows = len(self.data)
            changes = []

            # ทำความสะอาดตามตัวเลือกที่เลือก

            # 1. จัดการค่าว่าง
            fill_method = getattr(self, 'fill_method_var', None)
            if fill_method:
                missing_before = self.data.isnull().sum().sum()
                self.apply_missing_data_cleaning(fill_method.get())
                missing_after = self.data.isnull().sum().sum()
                if missing_before != missing_after:
                    changes.append(
                        f"เติมค่าว่าง: {missing_before - missing_after:,} ค่า")

            # 2. จัดการข้อมูลซ้ำ
            duplicate_method = getattr(self, 'duplicate_method_var', None)
            if duplicate_method and duplicate_method.get() != 'mark_only':
                dup_before = len(self.data)
                self.apply_duplicate_cleaning(duplicate_method.get())
                dup_after = len(self.data)
                if dup_before != dup_after:
                    changes.append(
                        f"ลบข้อมูลซ้ำ: {dup_before - dup_after:,} แถว")

            # 3. จัดการค่าผิดปกติ
            outlier_method = getattr(self, 'outlier_method_var', None)
            if outlier_method and outlier_method.get() != 'none':
                outlier_before = len(self.data)
                self.apply_outlier_cleaning(outlier_method.get())
                outlier_after = len(self.data)
                if outlier_before != outlier_after:
                    changes.append(
                        f"จัดการค่าผิดปกติ: {outlier_before - outlier_after:,} แถว")

            # 4. มาตรฐานข้อมูล
            if hasattr(self, 'text_options'):
                self.apply_standardization()
                changes.append("ปรับมาตรฐานข้อมูล")

            # อัปเดตการแสดงผล
            self.update_display()
            self.update_info_panel()            # แสดงผลลัพธ์
            if changes:
                result_text = "✅ ทำความสะอาดเสร็จสิ้น:\n" + \
                    "\n".join(f"• {change}" for change in changes)
            else:
                result_text = "ℹ️ ไม่มีการเปลี่ยนแปลงข้อมูล"
            messagebox.showinfo("เสร็จสิ้น", result_text)
            options_window.destroy()

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการทำความสะอาด:\n{str(e)}")

    def apply_missing_data_cleaning(self, method):
        """ดำเนินการเติมค่าว่างตามวิธีที่เลือก"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดไฟล์ข้อมูลก่อน")
            return

        if method == 'remove':
            self.data = self.data.dropna()
        elif method == 'custom':
            # ใช้ค่าที่กำหนดเองตามคอลัมน์
            if hasattr(self, 'custom_values'):
                for col, entry in self.custom_values.items():
                    custom_value = entry.get().strip()
                    if custom_value and col in self.data.columns:
                        try:
                            # แปลงประเภทข้อมูลให้เหมาะสม
                            if self.data[col].dtype in ['int64', 'float64']:
                                custom_value = float(custom_value)
                            self.data[col].fillna(custom_value, inplace=True)
                        except ValueError:
                            self.data[col].fillna(custom_value, inplace=True)
        else:            # วิธีมาตรฐาน
            for col in self.data.columns:
                if self.data[col].isnull().any():
                    if self.data[col].dtype in ['int64', 'float64']:
                        if method == 'mean':
                            self.data[col].fillna(
                                self.data[col].mean(), inplace=True)
                        elif method == 'median':
                            self.data[col].fillna(
                                self.data[col].median(), inplace=True)
                        elif method == 'zero':
                            self.data[col].fillna(0, inplace=True)
                        elif method == 'forward':
                            self.data[col] = self.data[col].ffill()
                        elif method == 'backward':
                            self.data[col] = self.data[col].bfill()
                        elif method == 'interpolate':
                            self.data[col].interpolate(inplace=True)
                    else:
                        if method == 'mode':
                            mode_val = self.data[col].mode()
                            if len(mode_val) > 0:
                                self.data[col].fillna(
                                    mode_val[0], inplace=True)
                        elif method == 'forward':
                            self.data[col] = self.data[col].ffill()
                        elif method == 'backward':
                            self.data[col] = self.data[col].bfill()

    def apply_duplicate_cleaning(self, method):
        """ดำเนินการลบข้อมูลซ้ำตามวิธีที่เลือก"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดไฟล์ข้อมูลก่อน")
            return

        if method == 'first':
            self.data = self.data.drop_duplicates(keep='first')
        elif method == 'last':
            self.data = self.data.drop_duplicates(keep='last')
        elif method == 'all':
            self.data = self.data.drop_duplicates(keep=False)

    def apply_outlier_cleaning(self, method):
        """ดำเนินการจัดการค่าผิดปกติ"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดไฟล์ข้อมูลก่อน")
            return

        action = getattr(self, 'outlier_action_var', None)
        if not action:
            return

        # Additional safety check to ensure data is not None
        if self.data is None:
            messagebox.showwarning("คำเตือน", "ข้อมูลไม่พร้อมใช้งาน")
            return

        # Check if data has numeric columns
        try:
            numeric_columns = self.data.select_dtypes(
                include=[np.number]).columns
            if len(numeric_columns) == 0:
                messagebox.showinfo(
                    "ข้อมูล", "ไม่พบคอลัมน์ตัวเลขสำหรับการตรวจจับค่าผิดปกติ")
                return
        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถวิเคราะห์ประเภทข้อมูลได้: {str(e)}")
            return

        for col in numeric_columns:
            # Additional safety check within the loop
            if self.data is None:
                messagebox.showwarning(
                    "คำเตือน", "ข้อมูลหายไประหว่างการประมวลผล")
                return

            try:
                if method == 'zscore':
                    threshold = getattr(self, 'zscore_threshold',
                                        tk.DoubleVar(value=3.0)).get()
                    z_scores = np.abs(
                        (self.data[col] - self.data[col].mean()) / self.data[col].std())
                    outliers = z_scores > threshold

                    if action.get() == 'remove':
                        self.data = self.data[~outliers]
                    elif action.get() == 'cap':
                        q1 = self.data[col].quantile(0.25)
                        q3 = self.data[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        self.data[col] = self.data[col].clip(
                            lower=lower_bound, upper=upper_bound)

                elif method == 'iqr':
                    q1 = self.data[col].quantile(0.25)
                    q3 = self.data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = (self.data[col] < lower_bound) | (
                        self.data[col] > upper_bound)

                    if action.get() == 'remove':
                        self.data = self.data[~outliers]
                    elif action.get() == 'cap':
                        self.data[col] = self.data[col].clip(
                            lower=lower_bound, upper=upper_bound)

            except Exception as e:
                messagebox.showerror(
                    "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการจัดการค่าผิดปกติสำหรับคอลัมน์ {col}: {str(e)}")
                continue

    def apply_standardization(self):
        """ดำเนินการปรับมาตรฐานข้อมูล"""
        if self.data is None:
            messagebox.showwarning("คำเตือน", "กรุณาโหลดไฟล์ข้อมูลก่อน")
            return

        if not hasattr(self, 'text_options'):
            return

        for col in self.data.select_dtypes(include=['object']).columns:
            # ลบช่องว่างที่ไม่จำเป็น
            if self.text_options.get('remove_whitespace', tk.BooleanVar()).get():
                self.data[col] = self.data[col].astype(str).str.strip()

            # ปรับตัวพิมพ์
            if self.text_options.get('standardize_case', tk.BooleanVar()).get():
                self.data[col] = self.data[col].astype(str).str.title()

            # ลบอักขระพิเศษ
            if self.text_options.get('remove_special_chars', tk.BooleanVar()).get():
                self.data[col] = self.data[col].astype(
                    str).str.replace(r'[^\w\s]', '', regex=True)

    def save_cleaning_template(self):
        """บันทึกการตั้งค่าการทำความสะอาดเป็นเทมเพลต"""
        template_name = simpledialog.askstring(
            "บันทึกเทมเพลต", "ใส่ชื่อเทมเพลต:")
        if not template_name:
            return

        template = {
            'name': template_name,
            'fill_method': getattr(self, 'fill_method_var', tk.StringVar()).get(),
            'duplicate_method': getattr(self, 'duplicate_method_var', tk.StringVar()).get(),
            'outlier_method': getattr(self, 'outlier_method_var', tk.StringVar()).get(),
            'outlier_threshold': getattr(self, 'zscore_threshold', tk.DoubleVar()).get(),
            'outlier_action': getattr(self, 'outlier_action_var', tk.StringVar()).get(),
            'created_date': datetime.now().isoformat()
        }

        # บันทึกเทมเพลต
        templates_file = 'cleaning_templates.json'
        try:
            if os.path.exists(templates_file):
                with open(templates_file, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
            else:
                templates = {}

            templates[template_name] = template

            with open(templates_file, 'w', encoding='utf-8') as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)

            messagebox.showinfo(
                "สำเร็จ", f"บันทึกเทมเพลต '{template_name}' เรียบร้อย")

        except Exception as e:
            messagebox.showerror(
                "ข้อผิดพลาด", f"ไม่สามารถบันทึกเทมเพลตได้:\n{str(e)}")

    def save_to_undo_stack(self):
        """บันทึกสถานะปัจจุบันลง undo stack"""
        if self.data is not None:
            self.undo_stack.append(self.data.copy())
            # จำกัดขนาด undo stack
            if len(self.undo_stack) > 10:
                self.undo_stack.pop(0)
            # ล้าง redo stack
            self.redo_stack.clear()

    def show_help(self):
        """แสดงคู่มือการใช้งาน"""
        help_window = tk.Toplevel(self.root)
        help_window.title("📖 คู่มือการใช้งาน")
        help_window.geometry("600x500")

        text_widget = tk.Text(help_window, wrap='word')
        scrollbar = ttk.Scrollbar(
            help_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        help_content = """📖 คู่มือการใช้งานระบบทำความสะอาดข้อมูล

🚀 การเริ่มต้น:
1. คลิก "📂 เปิดไฟล์" เพื่อเลือกไฟล์ข้อมูล (CSV หรือ Excel)
2. หรือใช้เมนู "ไฟล์ > นำเข้าตัวอย่าง" เพื่อทดลองกับข้อมูลตัวอย่าง

🧹 การทำความสะอาดข้อมูล:
• "🧹 ทำความสะอาดทั้งหมด" - ทำความสะอาดแบบครบวงจร
• "🗑️ ลบข้อมูลซ้ำ" - ลบแถวที่ซ้ำกัน
• "🔧 เติมค่าว่าง" - เติมค่าที่หายไป
• "🧼 ลบแถวว่าง" - ลบแถวที่ว่างเปล่า

⚙️ การตั้งค่า:
• ใช้เมนู "ตั้งค่า" เพื่อปรับแต่งวิธีการทำความสะอาด
• เลือกโหมดการทำงานที่เหมาะกับระดับของคุณ

📊 การตรวจสอบ:
• "📊 สถิติข้อมูล" - ดูข้อมูลสถิติโดยละเอียด
• "🎯 ตรวจสอบคุณภาพ" - ประเมินคุณภาพข้อมูล
• "📋 เปรียบเทียบ" - เปรียบเทียบก่อนและหลังการทำความสะอาด

💾 การบันทึก:
• "💾 บันทึก" - บันทึกข้อมูลที่ทำความสะอาดแล้ว
• "📄 ส่งออกรายงาน" - สร้างรายงานผลการทำความสะอาด

🎮 โหมดการทำงาน:
• ผู้เริ่มต้น: เหมาะสำหรับผู้ใช้ใหม่
• มาตรฐาน: โหมดปกติสำหรับผู้ใช้ทั่วไป  
• ผู้เชี่ยวชาญ: ตัวเลือกครบถ้วนสำหรับผู้เชี่ยวชาญ

❓ ต้องการความช่วยเหลือเพิ่มเติม?
• กดปุ่ม "💡 เคล็ดลับ" เพื่อดูเคล็ดลับการใช้งาน
• กดปุ่ม "🔍 ตัวอย่างการใช้งาน" เพื่อดูตัวอย่าง
"""

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        text_widget.insert('1.0', help_content)
        text_widget.config(state='disabled')

    def show_tips(self):
        """แสดงเคล็ดลับการใช้งาน"""
        tips = [
            "💡 สำรองข้อมูลต้นฉบับก่อนทำความสะอาดเสมอ",
            "🔍 ตรวจสอบข้อมูลก่อนและหลังการทำความสะอาด",
            "⚙️ ปรับการตั้งค่าให้เหมาะสมกับข้อมูลของคุณ",
            "📊 ใช้ฟังก์ชันสถิติเพื่อทำความเข้าใจข้อมูล",
            "🎯 ตรวจสอบค่าผิดปกติก่อนลบออก",
            "📄 สร้างรายงานเพื่อเก็บบันทึกการทำงาน"
        ]

        tip_text = "💡 เคล็ดลับการใช้งาน:\n\n" + "\n\n".join(tips)
        messagebox.showinfo("💡 เคล็ดลับ", tip_text)

    def show_examples(self):
        """แสดงตัวอย่างการใช้งาน"""
        examples = """🔍 ตัวอย่างการใช้งาน:

📝 สถานการณ์ที่ 1: ข้อมูลพนักงาน
• มีข้อมูลซ้ำ → ใช้ "ลบข้อมูลซ้ำ"
• มีช่องเงินเดือนว่าง → ใช้ "เติมค่าว่าง" (ค่าเฉลี่ย)
• มีอายุผิดปกติ (999) → ตรวจจับด้วย "ค่าผิดปกติ"

📊 สถานการณ์ที่ 2: ข้อมูลยอดขาย
• ตรวจสอบข้อมูลก่อน → "สถิติข้อมูล"
• ทำความสะอาดทั้งหมด → "ทำความสะอาดทั้งหมด"
• ตรวจสอบผลลัพธ์ → "เปรียบเทียบ"

🎯 สถานการณ์ที่ 3: ข้อมูลลูกค้า
• ลบแถวว่าง → "ลบแถวว่าง"
• มาตรฐานรูปแบบ → "มาตรฐานข้อมูล"
• ส่งออกรายงาน → "ส่งออกรายงาน"
"""
        messagebox.showinfo("🔍 ตัวอย่าง", examples)

    def show_about(self):
        """แสดงข้อมูลเกี่ยวกับโปรแกรม"""
        about_text = """🧹 ระบบทำความสะอาดข้อมูลแบบครบครัน 🇹🇭

เวอร์ชัน: 2.0
พัฒนาโดย: ไตรทศ ทองเกิด 097-191-2502

คุณสมบัติ:
✅ รองรับไฟล์ CSV และ Excel
✅ การทำความสะอาดข้อมูลแบบอัตโนมัติ
✅ ตรวจจับและจัดการค่าผิดปกติ
✅ รายงานผลการทำความสะอาดแบบละเอียด
✅ ระบบตั้งค่าที่ยืดหยุ่น
✅ รองรับภาษาไทยเต็มรูปแบบ

© 2025 All Rights Reserved
"""
        messagebox.showinfo("ℹ️ เกี่ยวกับ", about_text)


def main():
    root = tk.Tk()
    app = DataCleaningApp(root)

    # ตั้งค่าเริ่มต้น
    try:
        root.state('zoomed')  # เต็มจอใน Windows
    except:
        root.attributes('-zoomed', True)  # สำหรับ Linux

    # เพิ่มไอคอน (ถ้ามี)
    try:
        root.iconbitmap('icon.ico')
    except:
        pass

    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()


if __name__ == "__main__":
    main()

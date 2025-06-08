#!/usr/bin/env python3
"""
ระบบทำความสะอาดข้อมูล - อินเตอร์เฟซกราฟิก
==========================================

GUI สำหรับระบบทำความสะอาดข้อมูลที่ใช้งานง่ายด้วยภาษาไทย
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import json
from typing import Optional, List, Dict, Any, Union

# เพิ่ม modules path
sys.path.append(str(Path(__file__).parent))

from modules.data_loader import DataLoader
from modules.data_cleaner import DataCleaner
from modules.data_transformer import DataTransformer
from modules.data_validator import DataValidator
from modules.reporter import Reporter
from modules.utils import setup_logging, load_config

class DataCleansingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ระบบทำความสะอาดข้อมูล")
        self.root.geometry("1200x800")
        
        # ตั้งค่าธีม
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # โหลดการตั้งค่า
        try:
            self.config = load_config("config/config.yaml")
        except:
            self.config = {"database": {}, "processing": {"chunk_size": 1000}}
        
        # ตัวแปรสำหรับเก็บข้อมูล
        self.current_data: Optional[pd.DataFrame] = None
        self.original_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None
        self.cleaning_log: List[str] = []
        self.cleaned_results: List[str] = []
        
        # สร้าง instances
        self.data_loader = DataLoader(self.config)
        self.data_cleaner = DataCleaner()
        self.data_transformer = DataTransformer()
        self.data_validator = DataValidator()
        self.reporter = Reporter()
        
        # ตัวแปรเก็บประวัติการเปลี่ยนแปลง
        self.history: List[pd.DataFrame] = []
        self.history_index: int = -1
        
        # สร้าง UI
        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()
        
        # ตั้งค่า logging
        setup_logging()
    
    def get_current_data_info(self) -> Dict[str, Any]:
        """รับข้อมูลสถิติของข้อมูลปัจจุบัน"""
        info = {
            'rows': 0,
            'columns': 0,
            'missing_values': 0,
            'column_names': []
        }
        
        if self.current_data is not None:
            info.update({
                'rows': len(self.current_data),
                'columns': len(self.current_data.columns),
                'missing_values': self.current_data.isnull().sum().sum(),
                'column_names': list(self.current_data.columns)
            })
            
        return info
    
    def update_status(self, message: str):
        """อัพเดตข้อความในแถบสถานะ"""
        if hasattr(self, 'status_bar'):
            self.status_bar["text"] = message
    
    def create_menu(self):
        """สร้างเมนูหลัก"""
        menubar = tk.Menu(self.root)
        
        # เมนูไฟล์
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="นำเข้าข้อมูล", command=self.show_import_dialog)
        file_menu.add_command(label="บันทึก", command=self.save_data)
        file_menu.add_command(label="บันทึกเป็น...", command=self.save_data_as)
        file_menu.add_separator()
        file_menu.add_command(label="ออก", command=self.root.quit)
        menubar.add_cascade(label="ไฟล์", menu=file_menu)
        
        # เมนูแก้ไข
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="ยกเลิก", command=self.undo)
        edit_menu.add_command(label="ทำซ้ำ", command=self.redo)
        menubar.add_cascade(label="แก้ไข", menu=edit_menu)
        
        # เมนูมุมมอง
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="ข้อมูลดิบ", command=self.show_raw_data)
        view_menu.add_command(label="ข้อมูลที่ทำความสะอาดแล้ว", command=self.show_cleaned_data)
        view_menu.add_command(label="สถิติ", command=self.show_statistics)
        menubar.add_cascade(label="มุมมอง", menu=view_menu)
        
        # เมนูเครื่องมือ
        tools_menu = tk.Menu(menubar, tearoff=0)
        cleaning_menu = tk.Menu(tools_menu, tearoff=0)
        cleaning_menu.add_command(label="ลบแถวที่มีค่าว่าง", command=self.remove_missing)
        cleaning_menu.add_command(label="เติมค่าที่ขาด", command=self.fill_missing)
        cleaning_menu.add_command(label="ลบข้อมูลซ้ำ", command=self.remove_duplicates)
        cleaning_menu.add_command(label="แปลงรูปแบบข้อมูล", command=self.standardize_formats)
        cleaning_menu.add_command(label="ตรวจจับค่าผิดปกติ", command=self.detect_outliers)
        tools_menu.add_cascade(label="ทำความสะอาดข้อมูล", menu=cleaning_menu)
        
        tools_menu.add_separator()
        tools_menu.add_command(label="ตรวจสอบคุณภาพ", command=self.show_validation_dialog)
        tools_menu.add_command(label="สร้างรายงาน", command=self.show_report_dialog)
        menubar.add_cascade(label="เครื่องมือ", menu=tools_menu)
        
        # เมนูช่วยเหลือ
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="คู่มือการใช้งาน", command=self.show_help)
        help_menu.add_command(label="เกี่ยวกับ", command=self.show_about)
        menubar.add_cascade(label="ช่วยเหลือ", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_main_frame(self):
        """สร้างเฟรมหลัก"""
        # สร้าง Notebook สำหรับแท็บต่างๆ
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # แท็บข้อมูล
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="ข้อมูล")
        
        # สร้างตารางแสดงข้อมูล
        self.tree = ttk.Treeview(self.data_frame)
        self.tree.pack(fill='both', expand=True)
        
        # Scrollbar สำหรับตาราง
        self.vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.tree.yview)
        self.vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=self.vsb.set)
        
        # แท็บการทำความสะอาด
        self.cleaning_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.cleaning_frame, text="ทำความสะอาด")
        
        # ส่วนควบคุมการทำความสะอาด
        ttk.Label(self.cleaning_frame, text="ขั้นตอนการทำความสะอาด").pack(pady=5)
        
        # ปุ่มสำหรับขั้นตอนต่างๆ
        clean_buttons = [
            ("ลบแถวที่มีค่าว่าง", self.remove_missing),
            ("เติมค่าที่ขาด", self.fill_missing),
            ("ลบข้อมูลซ้ำ", self.remove_duplicates),
            ("แปลงรูปแบบข้อมูล", self.standardize_formats),
            ("ตรวจจับค่าผิดปกติ", self.detect_outliers)
        ]
        
        for text, command in clean_buttons:
            ttk.Button(self.cleaning_frame, text=text, command=command).pack(pady=2)
        
        # แท็บรายงาน
        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="รายงาน")
        
        ttk.Label(self.report_frame, text="รายงานผลการทำความสะอาด").pack(pady=5)
        self.report_text = tk.Text(self.report_frame)
        self.report_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """สร้างแถบสถานะ"""
        self.status_bar = ttk.Label(
            self.root, 
            text="พร้อมใช้งาน", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def show_import_dialog(self):
        """แสดงหน้าต่างนำเข้าข้อมูล"""
        filetypes = (
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        )
        
        import_dialog = tk.Toplevel(self.root)
        import_dialog.title("นำเข้าข้อมูล")
        import_dialog.geometry("500x300")
        
        # สร้างเฟรมสำหรับตัวเลือกการนำเข้า
        frame = ttk.Frame(import_dialog)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ตัวเลือกการนำเข้า
        ttk.Label(frame, text="เลือกแหล่งข้อมูล:").pack(anchor='w', pady=5)
        
        source_var = tk.StringVar(value="file")
        ttk.Radiobutton(frame, text="ไฟล์", variable=source_var, value="file").pack(anchor='w')
        ttk.Radiobutton(frame, text="ฐานข้อมูล", variable=source_var, value="database").pack(anchor='w')
        ttk.Radiobutton(frame, text="API", variable=source_var, value="api").pack(anchor='w')
        
        # เฟรมสำหรับการเลือกไฟล์
        file_frame = ttk.LabelFrame(frame, text="นำเข้าจากไฟล์", padding=10)
        file_frame.pack(fill='x', pady=10)
        
        file_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=file_path_var).pack(side='left', fill='x', expand=True)
        ttk.Button(file_frame, text="เลือกไฟล์", 
                  command=lambda: file_path_var.set(
                      filedialog.askopenfilename(
                          title="เลือกไฟล์ข้อมูล",
                          filetypes=filetypes
                      )
                  )).pack(side='right', padx=5)
        
        # เฟรมสำหรับฐานข้อมูล
        db_frame = ttk.LabelFrame(frame, text="เชื่อมต่อฐานข้อมูล", padding=10)
        db_frame.pack(fill='x', pady=10)
        
        ttk.Label(db_frame, text="Connection String:").pack(anchor='w')
        ttk.Entry(db_frame).pack(fill='x')
        
        # เฟรมสำหรับ API
        api_frame = ttk.LabelFrame(frame, text="เชื่อมต่อ API", padding=10)
        api_frame.pack(fill='x', pady=10)
        
        ttk.Label(api_frame, text="API Endpoint:").pack(anchor='w')
        ttk.Entry(api_frame).pack(fill='x')
        
        # ปุ่มดำเนินการ
        button_frame = ttk.Frame(import_dialog)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="นำเข้าข้อมูล", command=lambda: self._import_data(
            source_var.get(),
            file_path_var.get() if source_var.get() == "file" else None
        )).pack(side='right', padx=5)
        
        ttk.Button(button_frame, text="ยกเลิก", command=import_dialog.destroy).pack(side='right')
    
    def show_settings_dialog(self, setting_type: str):
        """แสดงหน้าต่างตั้งค่า"""
        settings_dialog = tk.Toplevel(self.root)
        settings_dialog.title("ตั้งค่า")
        settings_dialog.geometry("500x400")
        
        if setting_type == "data_standards":
            self._show_data_standards_settings(settings_dialog)
        elif setting_type == "user_permissions":
            self._show_user_permissions_settings(settings_dialog)
        elif setting_type == "connections":
            self._show_connection_settings(settings_dialog)
        elif setting_type == "language":
            self._show_language_settings(settings_dialog)
    
    def _show_data_standards_settings(self, parent):
        """แสดงการตั้งค่ามาตรฐานข้อมูล"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="มาตรฐานการตั้งชื่อคอลัมน์:").pack(anchor='w', pady=5)
        ttk.Entry(frame).pack(fill='x')
        
        ttk.Label(frame, text="รูปแบบวันที่:").pack(anchor='w', pady=5)
        ttk.Combobox(frame, values=["YYYY-MM-DD", "DD/MM/YYYY", "MM/DD/YYYY"]).pack(fill='x')
        
        ttk.Label(frame, text="การจัดการค่าว่าง:").pack(anchor='w', pady=5)
        ttk.Combobox(frame, values=["ลบแถว", "แทนที่ด้วยค่าเฉลี่ย", "แทนที่ด้วยค่ามัธยฐาน"]).pack(fill='x')
        
        ttk.Button(frame, text="บันทึก").pack(pady=10)
    
    def _show_user_permissions_settings(self, parent):
        """แสดงการตั้งค่าผู้ใช้และสิทธิ์"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ส่วนจัดการผู้ใช้
        users_frame = ttk.LabelFrame(frame, text="ผู้ใช้งาน", padding=10)
        users_frame.pack(fill='x', pady=5)
        
        # ตารางแสดงผู้ใช้
        users_tree = ttk.Treeview(users_frame, columns=("username", "role"), show="headings")
        users_tree.heading("username", text="ชื่อผู้ใช้")
        users_tree.heading("role", text="บทบาท")
        users_tree.pack(fill='x')
        
        # ปุ่มจัดการผู้ใช้
        button_frame = ttk.Frame(users_frame)
        button_frame.pack(fill='x', pady=5)
        
        ttk.Button(button_frame, text="เพิ่ม").pack(side='left', padx=5)
        ttk.Button(button_frame, text="แก้ไข").pack(side='left', padx=5)
        ttk.Button(button_frame, text="ลบ").pack(side='left', padx=5)
    
    def _show_connection_settings(self, parent):
        """แสดงการตั้งค่าการเชื่อมต่อ"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # การเชื่อมต่อฐานข้อมูล
        db_frame = ttk.LabelFrame(frame, text="ฐานข้อมูล", padding=10)
        db_frame.pack(fill='x', pady=5)
        
        ttk.Label(db_frame, text="Host:").pack(anchor='w')
        ttk.Entry(db_frame).pack(fill='x')
        
        ttk.Label(db_frame, text="Port:").pack(anchor='w')
        ttk.Entry(db_frame).pack(fill='x')
        
        ttk.Label(db_frame, text="Username:").pack(anchor='w')
        ttk.Entry(db_frame).pack(fill='x')
        
        ttk.Label(db_frame, text="Password:").pack(anchor='w')
        ttk.Entry(db_frame, show="*").pack(fill='x')
        
        # การเชื่อมต่อ API
        api_frame = ttk.LabelFrame(frame, text="API", padding=10)
        api_frame.pack(fill='x', pady=5)
        
        ttk.Label(api_frame, text="API Key:").pack(anchor='w')
        ttk.Entry(api_frame, show="*").pack(fill='x')
        
        ttk.Label(api_frame, text="Endpoint:").pack(anchor='w')
        ttk.Entry(api_frame).pack(fill='x')
        
        ttk.Button(frame, text="บันทึก").pack(pady=10)
    
    def _show_language_settings(self, parent):
        """แสดงการตั้งค่าภาษา"""
        frame = ttk.Frame(parent)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="เลือกภาษา:").pack(anchor='w', pady=5)
        
        lang_var = tk.StringVar(value="th")
        ttk.Radiobutton(frame, text="ไทย", variable=lang_var, value="th").pack(anchor='w')
        ttk.Radiobutton(frame, text="English", variable=lang_var, value="en").pack(anchor='w')
        
        ttk.Button(frame, text="บันทึก", 
                  command=lambda: self._save_language_settings(lang_var.get())).pack(pady=10)
    
    def _save_language_settings(self, language: str):
        """บันทึกการตั้งค่าภาษา"""
        # TODO: Implement language settings save
        messagebox.showinfo("แจ้งเตือน", f"บันทึกการตั้งค่าภาษาเป็น: {language}")
    
    def _import_data(self, source_type: str, file_path: Optional[str]):
        """ดำเนินการนำเข้าข้อมูล"""
        try:
            if source_type == "file" and file_path:
                # เก็บข้อมูลต้นฉบับ
                self.original_data = self.data_loader.load_data(file_path)
                self.current_data = self.original_data.copy()
                self.current_file = file_path
                
                # อัพเดตการแสดงผล
                self._update_data_view()
                self.status_bar["text"] = f"นำเข้าข้อมูลสำเร็จ: {file_path}"
                
            elif source_type == "database":
                # TODO: Implement database import
                messagebox.showinfo("แจ้งเตือน", "อยู่ระหว่างการพัฒนา: การนำเข้าจากฐานข้อมูล")
                
            elif source_type == "api":
                # TODO: Implement API import
                messagebox.showinfo("แจ้งเตือน", "อยู่ระหว่างการพัฒนา: การนำเข้าจาก API")
                
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถนำเข้าข้อมูลได้: {str(e)}")
    
    def _update_data_view(self):
        """อัพเดตมุมมองข้อมูลในตาราง"""
        if self.current_data is None:
            return
            
        # ล้างข้อมูลเก่า
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # กำหนดคอลัมน์
        columns = list(self.current_data.columns)
        self.tree["columns"] = columns
        
        # ตั้งค่าส่วนหัวคอลัมน์
        for col in columns:
            self.tree.heading(col, text=str(col))
            self.tree.column(col, width=100)
            
        # เพิ่มข้อมูล
        for idx, row in self.current_data.iterrows():
            values = list(row)
            self.tree.insert("", "end", text=str(idx), values=values)
    
    def _generate_report_preview(self, text_widget: tk.Text, report_type: str, export_type: str):
        """สร้างตัวอย่างรายงาน"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อมูลให้สร้างรายงาน")
            return
            
        text_widget.delete('1.0', tk.END)
        
        if report_type == "summary":
            # แสดงสรุปการทำความสะอาดข้อมูล
            text_widget.insert('end', "สรุปผลการทำความสะอาดข้อมูล\n")
            text_widget.insert('end', "=" * 40 + "\n\n")
            
            if self.original_data is not None:
                text_widget.insert('end', f"จำนวนข้อมูลต้นฉบับ: {len(self.original_data)} แถว\n")
            text_widget.insert('end', f"จำนวนข้อมูลที่ทำความสะอาด: {len(self.current_data)} แถว\n")
            
        elif report_type == "comparison":
            # แสดงการเปรียบเทียบก่อน-หลัง
            text_widget.insert('end', "เปรียบเทียบข้อมูลก่อน-หลังการทำความสะอาด\n")
            text_widget.insert('end', "=" * 40 + "\n\n")
            
            if self.original_data is not None and self.current_data is not None:
                for col in self.current_data.columns:
                    text_widget.insert('end', f"\nคอลัมน์: {col}\n")
                    text_widget.insert('end', f"ค่าที่ขาดหายไป (ก่อน): {self.original_data[col].isnull().sum()}\n")
                    text_widget.insert('end', f"ค่าที่ขาดหายไป (หลัง): {self.current_data[col].isnull().sum()}\n")
            
        elif report_type == "statistics":
            # แสดงสถิติข้อมูล
            text_widget.insert('end', "สถิติข้อมูล\n")
            text_widget.insert('end', "=" * 40 + "\n\n")
            
            for col in self.current_data.columns:
                text_widget.insert('end', f"\nคอลัมน์: {col}\n")
                if self.current_data[col].dtype in ['int64', 'float64']:
                    text_widget.insert('end', f"ค่าเฉลี่ย: {self.current_data[col].mean():.2f}\n")
                    text_widget.insert('end', f"ค่ามัธยฐาน: {self.current_data[col].median():.2f}\n")
                    text_widget.insert('end', f"ค่าต่ำสุด: {self.current_data[col].min():.2f}\n")
                    text_widget.insert('end', f"ค่าสูงสุด: {self.current_data[col].max():.2f}\n")
    
    def _save_report(self, report_type: str, export_type: str):
        """บันทึกรายงาน"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อมูลให้สร้างรายงาน")
            return
            
        try:
            # สร้างข้อมูลรายงาน
            report_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'report_type': report_type,
                'data_info': {
                    'original_rows': len(self.original_data) if self.original_data is not None else 0,
                    'cleaned_rows': len(self.current_data),
                    'columns': list(self.current_data.columns)
                }
            }
            
            if report_type == "summary":
                report_data['summary'] = self.data_cleaner.get_cleaning_summary(
                    self.original_data,
                    self.current_data,
                    self.cleaning_log
                )
            elif report_type == "comparison":
                if self.original_data is not None:
                    report_data['comparison'] = self.data_validator.validate_data(
                        self.original_data,
                        self.current_data
                    )
            elif report_type == "statistics":
                report_data['statistics'] = self.current_data.describe().to_dict()
        
            # บันทึกรายงาน
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                self.reporter.save_report(report_data, file_path)
                messagebox.showinfo("สำเร็จ", f"บันทึกรายงานเรียบร้อยแล้ว: {file_path}")
                
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถบันทึกรายงานได้: {str(e)}")
    
    def run(self):
        """เริ่มการทำงานของ GUI"""
        self.root.mainloop()

    def remove_missing(self):
        """ลบแถวที่มีข้อมูลขาดหายไป"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดข้อมูลก่อน")
            return

        try:
            # เก็บข้อมูลก่อนทำการเปลี่ยนแปลง
            self._save_state()
            
            # ลบแถวที่มีค่าว่าง
            self.current_data = self.data_cleaner._remove_empty_rows(self.current_data)
            self.current_data = self.data_cleaner._handle_missing_data(self.current_data)
            
            # อัพเดตการแสดงผล
            self._update_data_view()
            self.status_bar["text"] = "ลบแถวที่มีข้อมูลขาดหายไปเรียบร้อยแล้ว"
            
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถลบข้อมูลที่ขาดหายไปได้: {str(e)}")

    def fill_missing(self):
        """เติมค่าที่ขาดหายไป"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดข้อมูลก่อน")
            return

        try:
            # เก็บข้อมูลก่อนทำการเปลี่ยนแปลง
            self._save_state()
            
            # เติมค่าที่ขาดหายไป
            self.current_data = self.data_cleaner._handle_missing_data(self.current_data)
            
            # อัพเดตการแสดงผล
            self._update_data_view()
            self.status_bar["text"] = "เติมค่าที่ขาดหายไปเรียบร้อยแล้ว"
            
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถเติมค่าที่ขาดหายไปได้: {str(e)}")

    def remove_duplicates(self):
        """ลบข้อมูลซ้ำ"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดข้อมูลก่อน")
            return

        try:
            # เก็บข้อมูลก่อนทำการเปลี่ยนแปลง
            self._save_state()
            
            # ลบข้อมูลซ้ำ
            self.current_data = self.data_cleaner._remove_duplicates(self.current_data)
            
            # อัพเดตการแสดงผล
            self._update_data_view()
            self.status_bar["text"] = "ลบข้อมูลซ้ำเรียบร้อยแล้ว"
            
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถลบข้อมูลซ้ำได้: {str(e)}")

    def standardize_formats(self):
        """แปลงรูปแบบข้อมูลให้เป็นมาตรฐาน"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "กรุณาโหลดข้อมูลก่อน")
            return

        try:
            # เก็บข้อมูลก่อนทำการเปลี่ยนแปลง
            self._save_state()
            
            # แปลงรูปแบบข้อมูล
            self.current_data = self.data_cleaner._standardize_formats(self.current_data)
            
            # อัพเดตการแสดงผล
            self._update_data_view()
            self.status_bar["text"] = "แปลงรูปแบบข้อมูลเรียบร้อยแล้ว"
            
        except Exception as e:
            messagebox.showerror("เกิดข้อผิดพลาด", f"ไม่สามารถแปลงรูปแบบข้อมูลได้: {str(e)}")

    def _save_state(self):
        """บันทึกสถานะปัจจุบันเพื่อการ undo/redo"""
        if self.current_data is not None:
            if self.history_index < len(self.history) - 1:
                # ถ้ามีการ undo แล้วทำการแก้ไขใหม่ ให้ลบประวัติหลังจากตำแหน่งปัจจุบัน
                self.history = self.history[:self.history_index + 1]
            
            self.history.append(self.current_data.copy())
            self.history_index = len(self.history) - 1

    def show_raw_data(self):
        """แสดงข้อมูลดิบ"""
        if self.original_data is None:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อมูลดิบ")
            return
            
        data_window = tk.Toplevel(self.root)
        data_window.title("ข้อมูลดิบ")
        data_window.geometry("800x600")
        
        # สร้างตารางแสดงข้อมูล
        tree = ttk.Treeview(data_window)
        tree.pack(fill='both', expand=True)
        
        # Scrollbar
        vsb = ttk.Scrollbar(data_window, orient="vertical", command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)
        
        # กำหนดคอลัมน์
        columns = list(self.original_data.columns)
        tree["columns"] = columns
        
        for col in columns:
            tree.heading(col, text=str(col))
            tree.column(col, width=100)
            
        # เพิ่มข้อมูล
        for idx, row in self.original_data.iterrows():
            values = list(row)
            tree.insert("", "end", text=str(idx), values=values)
            
    def show_cleaned_data(self):
        """แสดงข้อมูลที่ทำความสะอาดแล้ว"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return
            
        data_window = tk.Toplevel(self.root)
        data_window.title("ข้อมูลที่ทำความสะอาดแล้ว")
        data_window.geometry("800x600")
        
        # สร้างตารางแสดงข้อมูล
        tree = ttk.Treeview(data_window)
        tree.pack(fill='both', expand=True)
        
        # Scrollbar
        vsb = ttk.Scrollbar(data_window, orient="vertical", command=tree.yview)
        vsb.pack(side='right', fill='y')
        tree.configure(yscrollcommand=vsb.set)
        
        # กำหนดคอลัมน์
        columns = list(self.current_data.columns)
        tree["columns"] = columns
        
        for col in columns:
            tree.heading(col, text=str(col))
            tree.column(col, width=100)
            
        # เพิ่มข้อมูล
        for idx, row in self.current_data.iterrows():
            values = list(row)
            tree.insert("", "end", text=str(idx), values=values)
            
    def show_statistics(self):
        """แสดงสถิติข้อมูล"""
        if self.current_data is None:
            messagebox.showwarning("แจ้งเตือน", "ไม่มีข้อมูลให้แสดงสถิติ")
            return
            
        stats_window = tk.Toplevel(self.root)
        stats_window.title("สถิติข้อมูล")
        stats_window.geometry("600x800")
        
        text_widget = tk.Text(stats_window)
        text_widget.pack(fill='both', expand=True)
        
        # แสดงสถิติพื้นฐาน
        desc = self.current_data.describe()
        text_widget.insert('end', "สถิติพื้นฐาน\n")
        text_widget.insert('end', "=" * 40 + "\n\n")
        text_widget.insert('end', str(desc))
        text_widget.insert('end', "\n\n")
        
        # แสดงข้อมูลเกี่ยวกับค่าที่ขาดหายไป
        text_widget.insert('end', "ข้อมูลค่าที่ขาดหายไป\n")
        text_widget.insert('end', "=" * 40 + "\n\n")
        for col in self.current_data.columns:
            missing = self.current_data[col].isnull().sum()
            text_widget.insert('end', f"{col}: {missing} ค่า\n")
        text_widget.insert('end', "\n")
        
        # แสดงประเภทข้อมูลของแต่ละคอลัมน์
        text_widget.insert('end', "ประเภทข้อมูล\n")
        text_widget.insert('end', "=" * 40 + "\n\n")
        for col in self.current_data.columns:
            dtype = self.current_data[col].dtype
            text_widget.insert('end', f"{col}: {dtype}\n")

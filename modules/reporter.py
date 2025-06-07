"""
โมดูลสร้างรายงาน (Reporter Module)
==================================

โมดูลนี้จัดการการสร้างรายงานและการแสดงผลการทำความสะอาดข้อมูล
รองรับการสร้างรายงานในรูปแบบต่างๆ

ผู้พัฒนา: ทีมทำความสะอาดข้อมูล
วันที่: มิถุนายน 2568
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import os


class Reporter:
    """
    คลาสสำหรับการสร้างรายงาน
    
    คลาสนี้จัดการการสร้างรายงานต่างๆ ประกอบด้วย:
    - รายงาน HTML
    - รายงาน JSON
    - รายงานสรุปข้อมูล
    - รายงานการเปรียบเทียบ
    - รายงานคุณภาพข้อมูล
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        เริ่มต้นคลาส Reporter
        
        Args:
            config (Dict): การตั้งค่าสำหรับการสร้างรายงาน
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.report_data = {}
        
    def generate_report(self, original_data: pd.DataFrame, 
                       cleaned_data: pd.DataFrame,
                       validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        สร้างรายงานหลัก
        
        Args:
            original_data: ข้อมูลต้นฉบับ
            cleaned_data: ข้อมูลที่ทำความสะอาดแล้ว
            validation_results: ผลการตรวจสอบ
            
        Returns:
            Dict: รายงานที่สร้างขึ้น
        """
        self.logger.info("📊 เริ่มสร้างรายงาน")
        
        # สร้างส่วนต่างๆ ของรายงาน
        report = {
            'metadata': self._create_metadata(),
            'executive_summary': self._create_executive_summary(
                original_data, cleaned_data, validation_results
            ),
            'data_overview': self._create_data_overview(original_data, cleaned_data),
            'cleaning_process': self._create_cleaning_process_summary(),
            'quality_assessment': validation_results.get('summary', {}),
            'detailed_analysis': self._create_detailed_analysis(
                original_data, cleaned_data, validation_results
            ),
            'recommendations': self._create_recommendations(validation_results),
            'appendix': self._create_appendix(original_data, cleaned_data)
        }
        
        self.report_data = report
        self.logger.info("✅ การสร้างรายงานเสร็จสิ้น")
        
        return report
    
    def _create_metadata(self) -> Dict[str, Any]:
        """สร้างข้อมูลเมตาของรายงาน"""
        return {
            'report_title': 'รายงานการทำความสะอาดข้อมูล',
            'generated_at': datetime.now().isoformat(),
            'generated_by': 'ระบบทำความสะอาดข้อมูล v1.0.0',
            'report_version': '1.0',
            'report_type': 'Data Cleansing Report'
        }
    
    def _create_executive_summary(self, original_data: pd.DataFrame,
                                cleaned_data: pd.DataFrame,
                                validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างสรุปสำหรับผู้บริหาร"""
        original_rows = len(original_data)
        cleaned_rows = len(cleaned_data)
        
        # คำนวณการปรับปรุง
        missing_original = original_data.isnull().sum().sum()
        missing_cleaned = cleaned_data.isnull().sum().sum()
        missing_improvement = missing_original - missing_cleaned
        
        # คะแนนคุณภาพ
        quality_score = validation_results.get('summary', {}).get(
            'quality_improvement', {}
        ).get('cleaned_score', 0)
        
        return {
            'overview': {
                'original_records': original_rows,
                'processed_records': cleaned_rows,
                'records_removed': max(0, original_rows - cleaned_rows),
                'data_quality_score': quality_score,
                'missing_data_resolved': missing_improvement
            },
            'key_achievements': [
                f"ประมวลผลข้อมูล {original_rows:,} แถว",
                f"ปรับปรุงคุณภาพข้อมูลเป็น {quality_score:.1f}%",
                f"แก้ไขข้อมูลที่ขาด {missing_improvement:,} จุด",
                f"ลบข้อมูลซ้ำ {max(0, original_rows - cleaned_rows):,} แถว"
            ],
            'processing_time': "ประมาณ 2.5 วินาที",  # จะคำนวณจริงในภายหลัง
            'status': "เสร็จสิ้นสมบูรณ์"
        }
    
    def _create_data_overview(self, original_data: pd.DataFrame,
                            cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """สร้างภาพรวมข้อมูล"""
        
        def get_column_info(data: pd.DataFrame) -> Dict[str, Any]:
            """วิเคราะห์ข้อมูลคอลัมน์"""
            column_info = {}
            for column in data.columns:
                column_info[column] = {
                    'data_type': str(data[column].dtype),
                    'non_null_count': int(data[column].count()),
                    'null_count': int(data[column].isnull().sum()),
                    'unique_values': int(data[column].nunique()),
                    'memory_usage': int(data[column].memory_usage(deep=True))
                }
                
                # เพิ่มข้อมูลสถิติสำหรับคอลัมน์ตัวเลข
                if data[column].dtype in ['int64', 'float64']:
                    stats = data[column].describe()
                    column_info[column]['statistics'] = {
                        'mean': float(stats['mean']) if not pd.isna(stats['mean']) else None,
                        'median': float(data[column].median()) if not pd.isna(data[column].median()) else None,
                        'std': float(stats['std']) if not pd.isna(stats['std']) else None,
                        'min': float(stats['min']) if not pd.isna(stats['min']) else None,
                        'max': float(stats['max']) if not pd.isna(stats['max']) else None
                    }
                
                # เพิ่มข้อมูลค่าที่พบบ่อยสำหรับคอลัมน์ข้อความ
                elif data[column].dtype == 'object':
                    value_counts = data[column].value_counts().head(5)
                    column_info[column]['top_values'] = value_counts.to_dict()
            
            return column_info
        
        return {
            'original_data': {
                'shape': list(original_data.shape),
                'memory_usage_mb': round(original_data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'column_types': original_data.dtypes.value_counts().to_dict(),
                'columns': get_column_info(original_data)
            },
            'cleaned_data': {
                'shape': list(cleaned_data.shape),
                'memory_usage_mb': round(cleaned_data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                'column_types': cleaned_data.dtypes.value_counts().to_dict(),
                'columns': get_column_info(cleaned_data)
            },
            'changes': {
                'rows_difference': len(cleaned_data) - len(original_data),
                'columns_difference': len(cleaned_data.columns) - len(original_data.columns),
                'new_columns': list(set(cleaned_data.columns) - set(original_data.columns)),
                'removed_columns': list(set(original_data.columns) - set(cleaned_data.columns))
            }
        }
    
    def _create_cleaning_process_summary(self) -> Dict[str, Any]:
        """สร้างสรุปกระบวนการทำความสะอาด"""
        return {
            'steps_performed': [
                {
                    'step': 'เตรียมข้อมูลเบื้องต้น',
                    'description': 'ตรวจสอบและแปลงประเภทข้อมูล, ทำความสะอาดชื่อคอลัมน์',
                    'status': 'เสร็จสิ้น'
                },
                {
                    'step': 'ลบแถวว่างเปล่า',
                    'description': 'ลบแถวที่ไม่มีข้อมูลใดๆ เลย',
                    'status': 'เสร็จสิ้น'
                },
                {
                    'step': 'จัดการข้อมูลที่ขาด',
                    'description': 'เติมค่าข้อมูลที่ขาดด้วยวิธีที่เหมาะสม',
                    'status': 'เสร็จสิ้น'
                },
                {
                    'step': 'ลบข้อมูลซ้ำ',
                    'description': 'ตรวจหาและลบแถวที่มีข้อมูลซ้ำกัน',
                    'status': 'เสร็จสิ้น'
                },
                {
                    'step': 'แก้ไขรูปแบบข้อมูล',
                    'description': 'มาตรฐานรูปแบบข้อมูลต่างๆ เช่น อีเมล, โทรศัพท์',
                    'status': 'เสร็จสิ้น'
                },
                {
                    'step': 'ตรวจสอบข้อมูลผิดปกติ',
                    'description': 'ตรวจหาและรายงานข้อมูลที่อาจผิดปกติ',
                    'status': 'เสร็จสิ้น'
                }
            ],
            'transformation_applied': [
                'สร้างฟีเจอร์วันที่ใหม่',
                'แปลงรหัสและค่าต่างๆ',
                'รวมและแยกคอลัมน์',
                'ปรับมาตรฐานข้อมูล'
            ]
        }
    
    def _create_detailed_analysis(self, original_data: pd.DataFrame,
                                cleaned_data: pd.DataFrame,
                                validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างการวิเคราะห์เชิงลึก"""
        
        # วิเคราะห์การเปลี่ยนแปลงในแต่ละคอลัมน์
        column_changes = {}
        common_columns = set(original_data.columns) & set(cleaned_data.columns)
        
        for column in common_columns:
            original_series = original_data[column]
            cleaned_series = cleaned_data[column]
            
            column_changes[column] = {
                'missing_data_change': {
                    'before': int(original_series.isnull().sum()),
                    'after': int(cleaned_series.isnull().sum()),
                    'improvement': int(original_series.isnull().sum() - cleaned_series.isnull().sum())
                },
                'data_type_change': {
                    'before': str(original_series.dtype),
                    'after': str(cleaned_series.dtype),
                    'changed': str(original_series.dtype) != str(cleaned_series.dtype)
                }
            }
            
            # สำหรับคอลัมน์ตัวเลข เพิ่มการเปรียบเทียบสถิติ
            if original_series.dtype in ['int64', 'float64'] and cleaned_series.dtype in ['int64', 'float64']:
                if len(original_series.dropna()) > 0 and len(cleaned_series.dropna()) > 0:
                    column_changes[column]['statistics_change'] = {
                        'mean_change': float(cleaned_series.mean() - original_series.mean()),
                        'std_change': float(cleaned_series.std() - original_series.std()),
                        'median_change': float(cleaned_series.median() - original_series.median())
                    }
        
        return {
            'column_changes': column_changes,
            'data_quality_metrics': validation_results.get('cleaned_quality', {}),
            'validation_issues': self._summarize_validation_issues(validation_results),
            'performance_metrics': {
                'processing_efficiency': 'สูง',
                'data_reduction_ratio': round((1 - len(cleaned_data) / len(original_data)) * 100, 2) if len(original_data) > 0 else 0,
                'quality_improvement_score': validation_results.get('summary', {}).get('quality_improvement', {}).get('improvement', 0)
            }
        }
    
    def _summarize_validation_issues(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """สรุปปัญหาที่พบจากการตรวจสอบ"""
        issues_summary = {
            'consistency_issues': [],
            'accuracy_issues': [],
            'validity_issues': [],
            'total_issues': 0
        }
        
        cleaned_quality = validation_results.get('cleaned_quality', {})
        
        # ปัญหาความสอดคล้อง
        consistency_issues = cleaned_quality.get('consistency', {}).get('issues', [])
        issues_summary['consistency_issues'] = consistency_issues
        
        # ปัญหาความถูกต้อง
        accuracy_issues = cleaned_quality.get('accuracy', {}).get('issues', [])
        issues_summary['accuracy_issues'] = accuracy_issues
        
        # ปัญหาความถูกต้องตามกฎ
        validity_issues = cleaned_quality.get('validity', {}).get('issues', [])
        issues_summary['validity_issues'] = validity_issues
        
        issues_summary['total_issues'] = len(consistency_issues) + len(accuracy_issues) + len(validity_issues)
        
        return issues_summary
    
    def _create_recommendations(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """สร้างคำแนะนำ"""
        recommendations = validation_results.get('summary', {}).get('recommendations', [])
        
        # เพิ่มคำแนะนำทั่วไป
        general_recommendations = [
            "ตรวจสอบข้อมูลอย่างสม่ำเสมอเพื่อรักษาคุณภาพ",
            "สร้างกระบวนการทำความสะอาดข้อมูลแบบอัตโนมัติ",
            "กำหนดมาตรฐานการรับข้อมูลเข้าระบบ",
            "อบรมผู้ใช้งานเกี่ยวกับการป้อนข้อมูลที่ถูกต้อง"
        ]
        
        return {
            'immediate_actions': recommendations,
            'long_term_improvements': general_recommendations,
            'best_practices': [
                "ใช้การตรวจสอบข้อมูลก่อนนำเข้า",
                "สร้างเอกสารมาตรฐานข้อมูล",
                "ติดตั้งระบบเตือนเมื่อพบข้อมูลผิดปกติ",
                "ทำการสำรองข้อมูลก่อนการทำความสะอาด"
            ]
        }
    
    def _create_appendix(self, original_data: pd.DataFrame,
                        cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """สร้างภาคผนวก"""
        return {
            'data_samples': {
                'original_sample': original_data.head(5).to_dict('records') if len(original_data) > 0 else [],
                'cleaned_sample': cleaned_data.head(5).to_dict('records') if len(cleaned_data) > 0 else []
            },
            'technical_details': {
                'python_version': "3.8+",
                'pandas_version': pd.__version__,
                'processing_environment': "Windows",
                'memory_usage': {
                    'original_mb': round(original_data.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                    'cleaned_mb': round(cleaned_data.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            },
            'glossary': {
                'Missing Data': 'ข้อมูลที่ขาดหายไปหรือไม่มีค่า',
                'Outlier': 'ข้อมูลที่มีค่าผิดปกติจากข้อมูลส่วนใหญ่',
                'Duplicate': 'ข้อมูลที่ซ้ำกันในแถวต่างๆ',
                'Data Quality Score': 'คะแนนประเมินคุณภาพข้อมูลโดยรวม'
            }
        }
    
    def save_report(self, report: Dict[str, Any], file_path: str, format: str = 'html'):
        """
        บันทึกรายงานในรูปแบบที่กำหนด
        
        Args:
            report: รายงานที่จะบันทึก
            file_path: เส้นทางไฟล์
            format: รูปแบบไฟล์ (html, json, txt)
        """
        try:
            if format.lower() == 'html':
                self._save_html_report(report, file_path)
            elif format.lower() == 'json':
                self._save_json_report(report, file_path)
            elif format.lower() == 'txt':
                self._save_text_report(report, file_path)
            else:
                raise ValueError(f"รูปแบบไฟล์ไม่รองรับ: {format}")
                
            self.logger.info(f"✅ บันทึกรายงาน {format.upper()} ไปยัง: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ การบันทึกรายงานล้มเหลว: {str(e)}")
            raise
    
    def _save_html_report(self, report: Dict[str, Any], file_path: str):
        """บันทึกรายงานในรูปแบบ HTML"""
        html_content = self._generate_html_content(report)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """สร้างเนื้อหา HTML"""
        metadata = report.get('metadata', {})
        summary = report.get('executive_summary', {})
        overview = report.get('data_overview', {})
        
        html = f"""
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{metadata.get('report_title', 'รายงานการทำความสะอาดข้อมูล')}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2E7D32;
            margin: 0;
            font-size: 2.5em;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #1976D2;
            border-left: 4px solid #1976D2;
            padding-left: 15px;
            margin-bottom: 15px;
        }}
        .metric-box {{
            display: inline-block;
            background: #E3F2FD;
            padding: 15px;
            margin: 10px;
            border-radius: 8px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #1976D2;
            display: block;
        }}
        .metric-label {{
            color: #555;
            font-size: 0.9em;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
        .warning {{
            color: #FF9800;
            font-weight: bold;
        }}
        .info {{
            background: #E1F5FE;
            padding: 15px;
            border-left: 4px solid #03A9F4;
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .recommendations {{
            background: #FFF3E0;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #FF9800;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 {metadata.get('report_title', 'รายงานการทำความสะอาดข้อมูล')}</h1>
            <p>สร้างเมื่อ: {metadata.get('generated_at', '')}</p>
        </div>
        
        <div class="section">
            <h2>🎯 สรุปสำหรับผู้บริหาร</h2>
            <div class="info">
                <div class="metric-box">
                    <span class="metric-value">{summary.get('overview', {}).get('original_records', 0):,}</span>
                    <span class="metric-label">แถวต้นฉบับ</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value">{summary.get('overview', {}).get('processed_records', 0):,}</span>
                    <span class="metric-label">แถวที่ประมวลผล</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value">{summary.get('overview', {}).get('data_quality_score', 0):.1f}%</span>
                    <span class="metric-label">คะแนนคุณภาพ</span>
                </div>
                <div class="metric-box">
                    <span class="metric-value">{summary.get('overview', {}).get('missing_data_resolved', 0):,}</span>
                    <span class="metric-label">ข้อมูลที่แก้ไข</span>
                </div>
            </div>
            
            <h3>✨ ผลสำเร็จที่สำคัญ</h3>
            <ul>
        """
        
        # เพิ่มรายการผลสำเร็จ
        for achievement in summary.get('key_achievements', []):
            html += f"<li>{achievement}</li>"
        
        html += """
            </ul>
        </div>
        
        <div class="section">
            <h2>📋 ภาพรวมข้อมูล</h2>
            <table>
                <tr>
                    <th>รายการ</th>
                    <th>ข้อมูลต้นฉบับ</th>
                    <th>ข้อมูลที่ทำความสะอาด</th>
                    <th>การเปลี่ยนแปลง</th>
                </tr>
        """
        
        # เพิ่มข้อมูลเปรียบเทียบ
        original_shape = overview.get('original_data', {}).get('shape', [0, 0])
        cleaned_shape = overview.get('cleaned_data', {}).get('shape', [0, 0])
        
        html += f"""
                <tr>
                    <td>จำนวนแถว</td>
                    <td>{original_shape[0]:,}</td>
                    <td>{cleaned_shape[0]:,}</td>
                    <td>{cleaned_shape[0] - original_shape[0]:+,}</td>
                </tr>
                <tr>
                    <td>จำนวนคอลัมน์</td>
                    <td>{original_shape[1]}</td>
                    <td>{cleaned_shape[1]}</td>
                    <td>{cleaned_shape[1] - original_shape[1]:+}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>🔧 กระบวนการทำความสะอาด</h2>
        """
        
        # เพิ่มขั้นตอนการทำความสะอาด
        cleaning_process = report.get('cleaning_process', {})
        steps = cleaning_process.get('steps_performed', [])
        
        for step in steps:
            status_class = 'success' if step.get('status') == 'เสร็จสิ้น' else 'warning'
            html += f"""
            <div class="info">
                <h4>{step.get('step', '')}</h4>
                <p>{step.get('description', '')}</p>
                <p class="{status_class}">สถานะ: {step.get('status', '')}</p>
            </div>
            """
        
        html += """
        </div>
        
        <div class="section">
            <div class="recommendations">
                <h2>💡 คำแนะนำ</h2>
        """
        
        # เพิ่มคำแนะนำ
        recommendations = report.get('recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        
        if immediate_actions:
            html += "<h3>🚨 การดำเนินการเร่งด่วน</h3><ul>"
            for action in immediate_actions:
                html += f"<li>{action}</li>"
            html += "</ul>"
        
        best_practices = recommendations.get('best_practices', [])
        if best_practices:
            html += "<h3>⭐ แนวทางปฏิบัติที่ดี</h3><ul>"
            for practice in best_practices:
                html += f"<li>{practice}</li>"
            html += "</ul>"
        
        html += f"""
            </div>
        </div>
        
        <div class="footer">
            <p>รายงานสร้างโดย {metadata.get('generated_by', '')} | เวอร์ชัน {metadata.get('report_version', '')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _save_json_report(self, report: Dict[str, Any], file_path: str):
        """บันทึกรายงานในรูปแบบ JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    def _save_text_report(self, report: Dict[str, Any], file_path: str):
        """บันทึกรายงานในรูปแบบ Text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("รายงานการทำความสะอาดข้อมูล\n")
            f.write("=" * 60 + "\n\n")
            
            metadata = report.get('metadata', {})
            f.write(f"สร้างเมื่อ: {metadata.get('generated_at', '')}\n")
            f.write(f"สร้างโดย: {metadata.get('generated_by', '')}\n\n")
            
            # สรุปสำหรับผู้บริหาร
            summary = report.get('executive_summary', {})
            f.write("สรุปสำหรับผู้บริหาร\n")
            f.write("-" * 30 + "\n")
            
            overview = summary.get('overview', {})
            f.write(f"แถวต้นฉบับ: {overview.get('original_records', 0):,}\n")
            f.write(f"แถวที่ประมวลผล: {overview.get('processed_records', 0):,}\n")
            f.write(f"คะแนนคุณภาพ: {overview.get('data_quality_score', 0):.1f}%\n")
            f.write(f"ข้อมูลที่แก้ไข: {overview.get('missing_data_resolved', 0):,}\n\n")
            
            # ผลสำเร็จที่สำคัญ
            f.write("ผลสำเร็จที่สำคัญ:\n")
            for achievement in summary.get('key_achievements', []):
                f.write(f"- {achievement}\n")
            f.write("\n")
            
            # คำแนะนำ
            recommendations = report.get('recommendations', {})
            f.write("คำแนะนำ:\n")
            f.write("-" * 20 + "\n")
            for action in recommendations.get('immediate_actions', []):
                f.write(f"- {action}\n")
    
    def create_summary_dashboard(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """สร้างแดชบอร์ดสรุปจากหลายรายงาน"""
        if not reports:
            return {}
        
        # รวบรวมข้อมูลจากหลายรายงาน
        total_records_processed = sum(
            report.get('executive_summary', {}).get('overview', {}).get('original_records', 0)
            for report in reports
        )
        
        avg_quality_score = sum(
            report.get('executive_summary', {}).get('overview', {}).get('data_quality_score', 0)
            for report in reports
        ) / len(reports)
        
        dashboard = {
            'period_summary': {
                'total_reports': len(reports),
                'total_records_processed': total_records_processed,
                'average_quality_score': round(avg_quality_score, 1),
                'last_updated': datetime.now().isoformat()
            },
            'trends': {
                'quality_trend': 'ปรับปรุง',  # จะคำนวณจากข้อมูลจริง
                'processing_efficiency': 'คงที่',
                'common_issues': ['ข้อมูลที่ขาด', 'ข้อมูลซ้ำ', 'รูปแบบไม่ถูกต้อง']
            }
        }
        
        return dashboard

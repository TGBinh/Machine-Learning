"""
Views xử lý các yêu cầu HTTP cho ứng dụng phân tích tài chính
Bao gồm upload file, phân tích dữ liệu và hiển thị kết quả
"""
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import pandas as pd
import json
import os
import tempfile
from typing import Dict, Any, Optional, List
import logging

# Import các tiện ích tự tạo
from .utils.fraud_detection import FraudDetectionKMeans
from .utils.personal_finance import PersonalFinanceAnalyzer

# Thiết lập logging
logger = logging.getLogger(__name__)

# Khởi tạo các analyzer
fraud_detector = FraudDetectionKMeans()
personal_finance_analyzer = PersonalFinanceAnalyzer()

def home(request):
    """
    Trang chủ mới với hai lựa chọn
    """
    return render(request, 'transactions/home.html')

def fraud_detection(request):
    """
    Trang phát hiện gian lận
    """
    return render(request, 'transactions/fraud_detection.html')

def personal_finance(request):
    """
    Trang phân tích tài chính cá nhân
    """
    return render(request, 'transactions/personal_finance.html')

@csrf_exempt
@require_http_methods(["POST"])
def upload_fraud_detection(request):
    """
    Xử lý upload file cho phát hiện gian lận
    """
    try:
        # Kiểm tra file được upload
        if 'file_excel' not in request.FILES:
            return JsonResponse({
                'thanh_cong': False,
                'loi': 'Không tìm thấy file được upload'
            })
        
        file_excel = request.FILES['file_excel']
        
        # Kiểm tra định dạng file
        if not file_excel.name.endswith('.xlsx'):
            return JsonResponse({
                'thanh_cong': False,
                'loi': 'Chỉ chấp nhận file Excel (.xlsx)'
            })
        
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as file_tam:
            for chunk in file_excel.chunks():
                file_tam.write(chunk)
            duong_dan_file_tam = file_tam.name
        
        try:
            # Đọc dữ liệu
            df = pd.read_excel(duong_dan_file_tam)
            logger.info(f"Loaded fraud detection data with {len(df)} rows and columns: {list(df.columns)}")
            
            # Thực hiện phát hiện gian lận với pre-trained model
            results = fraud_detector.detect_fraud_pretrained(df)
            
            return JsonResponse({
                'thanh_cong': True,
                'thong_bao': f'Phát hiện {results["suspicious_count"]} giao dịch nghi ngờ',
                'results': results,
                'elbow_data': results['elbow_data'],
                'cluster_data': results['cluster_info'],
                'scatter_data': results['scatter_data']
            })
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích gian lận: {str(e)}")
            return JsonResponse({
                'thanh_cong': False,
                'loi': f'Lỗi khi phân tích dữ liệu: {str(e)}'
            })
        
        finally:
            # Xóa file tạm thời
            try:
                os.unlink(duong_dan_file_tam)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Lỗi upload file gian lận: {str(e)}")
        return JsonResponse({
            'thanh_cong': False,
            'loi': f'Lỗi hệ thống: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def upload_personal_finance(request):
    """
    Xử lý upload file cho phân tích tài chính cá nhân
    """
    try:
        # Kiểm tra file được upload
        if 'file_excel' not in request.FILES:
            return JsonResponse({
                'thanh_cong': False,
                'loi': 'Không tìm thấy file được upload'
            })
        
        file_excel = request.FILES['file_excel']
        
        # Kiểm tra định dạng file
        if not file_excel.name.endswith('.xlsx'):
            return JsonResponse({
                'thanh_cong': False,
                'loi': 'Chỉ chấp nhận file Excel (.xlsx)'
            })
        
        # Lưu file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as file_tam:
            for chunk in file_excel.chunks():
                file_tam.write(chunk)
            duong_dan_file_tam = file_tam.name
        
        try:
            # Đọc dữ liệu
            df = pd.read_excel(duong_dan_file_tam)
            logger.info(f"Loaded personal finance data with {len(df)} rows and columns: {list(df.columns)}")
            
            # Kiểm tra dữ liệu cơ bản
            if df.empty:
                return JsonResponse({
                    'thanh_cong': False,
                    'loi': 'File Excel trống hoặc không có dữ liệu'
                })
            
            # Thực hiện phân tích tài chính cá nhân với timeout protection
            import threading
            import time
            
            results = None
            error_occurred = None
            analysis_completed = threading.Event()
            
            def run_analysis():
                nonlocal results, error_occurred
                try:
                    results = personal_finance_analyzer.analyze_personal_finance_pretrained(df)
                except Exception as e:
                    error_occurred = e
                finally:
                    analysis_completed.set()
            
            # Chạy analysis trong thread riêng
            analysis_thread = threading.Thread(target=run_analysis)
            analysis_thread.daemon = True
            analysis_thread.start()
            
            # Chờ tối đa 8 giây
            if analysis_completed.wait(timeout=8):
                # Analysis completed
                if error_occurred:
                    raise error_occurred
                
                return JsonResponse({
                    'thanh_cong': True,
                    'thong_bao': 'Phân tích tài chính cá nhân hoàn tất',
                    'results': results
                })
            else:
                # Timeout occurred
                logger.error("Personal finance analysis timed out")
                return JsonResponse({
                    'thanh_cong': False,
                    'loi': 'Quá trình phân tích mất quá nhiều thời gian. Vui lòng thử với file nhỏ hơn.'
                })
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích tài chính cá nhân: {str(e)}")
            return JsonResponse({
                'thanh_cong': False,
                'loi': f'Lỗi khi phân tích dữ liệu: {str(e)}'
            })
        
        finally:
            # Xóa file tạm thời
            try:
                os.unlink(duong_dan_file_tam)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Lỗi upload file tài chính cá nhân: {str(e)}")
        return JsonResponse({
            'thanh_cong': False,
            'loi': f'Lỗi hệ thống: {str(e)}'
        })

@csrf_exempt
@require_http_methods(["POST"])
def download_personal_report(request):
    """
    Tạo và tải báo cáo PDF cho phân tích tài chính cá nhân
    """
    try:
        import threading
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        # Đọc dữ liệu phân tích từ request với timeout
        pdf_buffer = None
        error_occurred = None
        pdf_completed = threading.Event()
        
        def create_pdf():
            nonlocal pdf_buffer, error_occurred
            try:
                if request.content_type == 'application/json':
                    import json
                    analysis_data = json.loads(request.body)
                else:
                    raise ValueError('Dữ liệu không hợp lệ')
                
                # Tạo PDF buffer
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                
                # Tạo custom style cho tiếng Việt
                title_style = ParagraphStyle(
                    'CustomTitle',
                    parent=styles['Heading1'],
                    fontSize=18,
                    spaceAfter=30,
                    alignment=1  # Center
                )
                
                heading_style = ParagraphStyle(
                    'CustomHeading',
                    parent=styles['Heading2'],
                    fontSize=14,
                    spaceAfter=12
                )
                
                # Story list để chứa nội dung PDF
                story = []
                
                # Title
                story.append(Paragraph("BÁO CÁO TÀI CHÍNH CÁ NHÂN", title_style))
                story.append(Spacer(1, 20))
                
                # Tổng quan
                story.append(Paragraph("TỔNG QUAN", heading_style))
                overview = analysis_data.get('overview', {})
                overview_data = [
                    ["Chỉ số", "Giá trị"],
                    ["Tổng chi tiêu", f"{overview.get('total_spending', 'N/A')} VND"],
                    ["Số giao dịch", f"{overview.get('total_transactions', 'N/A')}"],
                    ["Chi tiêu TB/ngày", f"{overview.get('avg_daily_spending', 'N/A')} VND"],
                ]
                
                overview_table = Table(overview_data, colWidths=[2*inch, 3*inch])
                overview_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(overview_table)
                story.append(Spacer(1, 20))
                
                # Phân loại chi tiêu
                story.append(Paragraph("PHÂN LOẠI CHI TIÊU", heading_style))
                categories = analysis_data.get('categories', {}).get('details', [])
                category_data = [["Danh mục", "Số tiền (VND)", "Tỷ lệ (%)"]]
                
                for cat in categories:
                    category_data.append([
                        cat.get('name', ''),
                        f"{cat.get('amount', 0):,.0f}",
                        f"{cat.get('percentage', 0):.1f}"
                    ])
                
                category_table = Table(category_data, colWidths=[2*inch, 2*inch, 1*inch])
                category_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(category_table)
                story.append(Spacer(1, 20))
                
                # Gợi ý tiết kiệm
                story.append(Paragraph("GỢI Ý TIẾT KIỆM", heading_style))
                savings = analysis_data.get('savings', {})
                savings_data = [
                    ["Chỉ số", "Giá trị"],
                    ["Mục tiêu tiết kiệm", f"{savings.get('target', 'N/A')} VND/tháng"],
                    ["Có thể tiết kiệm", f"{savings.get('potential', 'N/A')} VND/tháng"],
                    ["Tỷ lệ đạt được", f"{savings.get('achievable_rate', 'N/A')}%"]
                ]
                
                savings_table = Table(savings_data, colWidths=[2*inch, 3*inch])
                savings_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(savings_table)
                story.append(Spacer(1, 20))
                
                # Footer
                story.append(Paragraph("Báo cáo được tạo tự động bởi hệ thống phân tích tài chính.", styles['Normal']))
                
                # Build PDF
                doc.build(story)
                pdf_buffer = buffer.getvalue()
                buffer.close()
                
            except Exception as e:
                error_occurred = e
            finally:
                pdf_completed.set()
        
        # Chạy PDF generation trong thread riêng với timeout
        pdf_thread = threading.Thread(target=create_pdf)
        pdf_thread.daemon = True
        pdf_thread.start()
        
        # Chờ tối đa 8 giây
        if pdf_completed.wait(timeout=8):
            if error_occurred:
                logger.error(f"Error creating PDF: {str(error_occurred)}")
                return JsonResponse({
                    'thanh_cong': False,
                    'loi': f'Lỗi khi tạo báo cáo: {str(error_occurred)}'
                })
            
            # Trả về PDF
            response = HttpResponse(pdf_buffer, content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="bao_cao_tai_chinh_ca_nhan.pdf"'
            return response
        else:
            # Timeout
            logger.error("PDF creation timed out")
            return JsonResponse({
                'thanh_cong': False,
                'loi': 'Tạo báo cáo mất quá nhiều thời gian. Vui lòng thử lại.'
            })
        
    except Exception as e:
        logger.error(f"Lỗi tạo báo cáo: {str(e)}")
        return JsonResponse({
            'thanh_cong': False,
            'loi': f'Lỗi khi tạo báo cáo: {str(e)}'
        })
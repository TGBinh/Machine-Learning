"""
URL patterns cho ứng dụng phân tích giao dịch tài chính
"""
from django.urls import path
from . import views

app_name = 'transactions'

urlpatterns = [
    # Trang chủ
    path('', views.home, name='home'),
    
    # Các trang chức năng
    path('fraud-detection/', views.fraud_detection, name='fraud_detection'),
    path('personal-finance/', views.personal_finance, name='personal_finance'),
    
    # Upload endpoints
    path('upload-fraud/', views.upload_fraud_detection, name='upload_fraud_detection'),
    path('upload-personal/', views.upload_personal_finance, name='upload_personal_finance'),
    path('download-report/', views.download_personal_report, name='download_personal_report'),
]
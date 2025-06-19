"""
Lệnh Django để import dữ liệu mẫu từ file Excel vào cơ sở dữ liệu
"""
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from transactions.models import Transaction
from transactions.utils.data_processor import BoXuLyDuLieu
import pandas as pd
import os
from datetime import datetime
import random

class Command(BaseCommand):
    help = 'Import dữ liệu giao dịch mẫu từ file Excel vào cơ sở dữ liệu SQLite'

    def add_arguments(self, parser):
        parser.add_argument(
            '--file',
            type=str,
            help='Đường dẫn đến file Excel chứa dữ liệu',
            default='data.xlsx'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Xóa dữ liệu cũ trước khi import',
        )
        parser.add_argument(
            '--sample',
            action='store_true',
            help='Tạo dữ liệu mẫu nếu không có file Excel',
        )

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('=== BẮT ĐẦU IMPORT DỮ LIỆU ==='))
        
        file_path = options['file']
        
        # Kiểm tra file có tồn tại không
        if not os.path.exists(file_path):
            if options['sample']:
                self.stdout.write(self.style.WARNING(f'Không tìm thấy file {file_path}. Tạo dữ liệu mẫu...'))
                # Xóa dữ liệu cũ trước khi tạo sample data
                if Transaction.objects.exists():
                    self.stdout.write('Đang xóa dữ liệu cũ trước khi tạo dữ liệu mẫu...')
                    Transaction.objects.all().delete()
                    self.stdout.write(self.style.SUCCESS('Đã xóa dữ liệu cũ'))
                self.tao_du_lieu_mau()
                return
            else:
                raise CommandError(f'Không tìm thấy file: {file_path}')
        
        # Nếu có file thật và có dữ liệu cũ, xóa dữ liệu cũ
        if Transaction.objects.exists():
            self.stdout.write('Phát hiện file dữ liệu thật. Đang xóa dữ liệu mẫu cũ...')
            Transaction.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Đã xóa dữ liệu cũ, sẵn sàng import dữ liệu thật'))
        
        # Xóa dữ liệu cũ nếu được yêu cầu
        if options['clear']:
            self.stdout.write('Đang xóa dữ liệu cũ theo yêu cầu...')
            Transaction.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('Đã xóa dữ liệu cũ'))

        try:
            # Khởi tạo bộ xử lý dữ liệu
            bo_xu_ly = BoXuLyDuLieu()
            
            # Đọc và xử lý dữ liệu
            self.stdout.write('Đang đọc file Excel...')
            df_goc = bo_xu_ly.doc_file_excel(file_path)
            
            self.stdout.write('Đang chuẩn hóa dữ liệu...')
            df_chuan_hoa = bo_xu_ly.chuan_hoa_ten_cot(df_goc)
            df_sach = bo_xu_ly.lam_sach_du_lieu(df_chuan_hoa)
            
            # Import vào database
            self.stdout.write('Đang import vào cơ sở dữ liệu...')
            so_luong_import = self.import_vao_database(df_sach)
            
            # Tạo tóm tắt
            tom_tat = bo_xu_ly.tao_tom_tat_du_lieu(df_sach)
            self.hien_thi_tom_tat(tom_tat, so_luong_import)
            
            self.stdout.write(
                self.style.SUCCESS(f'✅ HOÀN THÀNH! Đã import {so_luong_import} giao dịch')
            )
            
        except Exception as e:
            raise CommandError(f'Lỗi khi import dữ liệu: {str(e)}')

    def import_vao_database(self, df):
        """
        Import DataFrame vào cơ sở dữ liệu
        """
        so_luong_import = 0
        so_luong_loi = 0
        
        for index, row in df.iterrows():
            try:
                # Chuyển đổi dữ liệu
                giao_dich_data = self.chuyen_doi_du_lieu_row(row)
                
                # Tạo hoặc cập nhật giao dịch
                giao_dich, created = Transaction.objects.get_or_create(
                    ma_giao_dich=giao_dich_data['ma_giao_dich'],
                    defaults=giao_dich_data
                )
                
                if created:
                    so_luong_import += 1
                    if so_luong_import % 100 == 0:
                        self.stdout.write(f'Đã import {so_luong_import} giao dịch...')
                        
            except Exception as e:
                so_luong_loi += 1
                if so_luong_loi <= 5:  # Chỉ hiển thị 5 lỗi đầu tiên
                    self.stdout.write(
                        self.style.WARNING(f'Lỗi tại dòng {index + 1}: {str(e)}')
                    )
        
        if so_luong_loi > 0:
            self.stdout.write(
                self.style.WARNING(f'Có {so_luong_loi} dòng bị lỗi không thể import')
            )
        
        return so_luong_import

    def chuyen_doi_du_lieu_row(self, row):
        """
        Chuyển đổi một dòng dữ liệu thành format phù hợp với model
        """
        return {
            'ma_giao_dich': str(row.get('ma_giao_dich', f'GD_{random.randint(100000, 999999)}')),
            'id_nguoi_gui': str(row.get('id_nguoi_gui', f'USER_{random.randint(1000, 9999)}')),
            'id_nguoi_nhan': str(row.get('id_nguoi_nhan', f'USER_{random.randint(1000, 9999)}')),
            'so_tien': float(row.get('so_tien', 0)),
            'so_du_tai_khoan': float(row.get('so_du_tai_khoan', 0)),
            'ngay_gio_giao_dich': pd.to_datetime(row.get('ngay_gio_giao_dich')),
            'loai_giao_dich': str(row.get('loai_giao_dich', 'KHAC')),
            'trang_thai_giao_dich': str(row.get('trang_thai_giao_dich', 'HOAN_THANH')),
            'noi_dung_giao_dich': str(row.get('noi_dung_giao_dich', 'Không có mô tả')),
            'co_gian_lan': bool(row.get('co_gian_lan', False))
        }

    def tao_du_lieu_mau(self):
        """
        Tạo dữ liệu mẫu để test
        """
        from datetime import datetime, timedelta
        import random
        
        self.stdout.write('Đang tạo dữ liệu mẫu...')
        
        # Danh sách các loại giao dịch
        cac_loai_giao_dich = ['CHUYEN_KHOAN', 'THANH_TOAN', 'RUT_TIEN', 'GUI_TIEN', 'MUA_SAM']
        cac_trang_thai = ['HOAN_THANH', 'CHO_XU_LY', 'THAT_BAI']
        cac_noi_dung = [
            'Chuyển tiền cho bạn', 'Thanh toán hóa đơn', 'Mua sắm online',
            'Nạp tiền điện thoại', 'Thanh toán dịch vụ', 'Chuyển tiền gia đình',
            'Mua thực phẩm', 'Thanh toán tiền điện', 'Đóng học phí'
        ]
        
        # Tạo 1000 giao dịch mẫu
        so_luong_tao = 0
        
        for i in range(1000):
            try:
                # Tạo dữ liệu ngẫu nhiên
                ngay_giao_dich = datetime.now() - timedelta(days=random.randint(1, 365))
                so_tien = random.uniform(10000, 5000000)  # 10k - 5M VND
                
                giao_dich = Transaction.objects.create(
                    ma_giao_dich=f'GD_{i+1:06d}',
                    id_nguoi_gui=f'USER_{random.randint(1001, 1050)}',  # 50 người dùng
                    id_nguoi_nhan=f'USER_{random.randint(1001, 1050)}',
                    so_tien=so_tien,
                    so_du_tai_khoan=random.uniform(100000, 10000000),
                    ngay_gio_giao_dich=ngay_giao_dich,
                    loai_giao_dich=random.choice(cac_loai_giao_dich),
                    trang_thai_giao_dich=random.choice(cac_trang_thai),
                    noi_dung_giao_dich=random.choice(cac_noi_dung),
                    co_gian_lan=random.random() < 0.05  # 5% giao dịch gian lận
                )
                
                so_luong_tao += 1
                
                if so_luong_tao % 100 == 0:
                    self.stdout.write(f'Đã tạo {so_luong_tao} giao dịch mẫu...')
                    
            except Exception as e:
                self.stdout.write(
                    self.style.WARNING(f'Lỗi tạo giao dịch {i+1}: {str(e)}')
                )
        
        self.stdout.write(
            self.style.SUCCESS(f'✅ Đã tạo {so_luong_tao} giao dịch mẫu')
        )

    def hien_thi_tom_tat(self, tom_tat, so_luong_import):
        """
        Hiển thị tóm tắt dữ liệu đã import
        """
        self.stdout.write('\n' + '='*50)
        self.stdout.write(self.style.SUCCESS('📊 TÓM TẮT DỮ LIỆU ĐÃ IMPORT'))
        self.stdout.write('='*50)
        
        self.stdout.write(f'🔢 Tổng số giao dịch: {tom_tat.get("tong_so_giao_dich", 0):,}')
        self.stdout.write(f'✅ Số giao dịch đã import: {so_luong_import:,}')
        
        thong_ke_tien = tom_tat.get('thong_ke_so_tien', {})
        self.stdout.write(f'💰 Tổng giá trị: {thong_ke_tien.get("tong_cong", 0):,.0f} VND')
        self.stdout.write(f'📊 Giá trị trung bình: {thong_ke_tien.get("trung_binh", 0):,.0f} VND')
        self.stdout.write(f'📈 Giá trị cao nhất: {thong_ke_tien.get("gia_tri_lon_nhat", 0):,.0f} VND')
        self.stdout.write(f'📉 Giá trị thấp nhất: {thong_ke_tien.get("gia_tri_nho_nhat", 0):,.0f} VND')
        
        thong_ke_gian_lan = tom_tat.get('thong_ke_gian_lan', {})
        self.stdout.write(f'⚠️  Số giao dịch gian lận: {thong_ke_gian_lan.get("tong_so_gian_lan", 0):,}')
        self.stdout.write(f'📊 Tỷ lệ gian lận: {thong_ke_gian_lan.get("ty_le_gian_lan", 0)*100:.2f}%')
        
        khoang_thoi_gian = tom_tat.get('khoang_thoi_gian', {})
        if khoang_thoi_gian.get('bat_dau') and khoang_thoi_gian.get('ket_thuc'):
            self.stdout.write(f'📅 Khoảng thời gian: {khoang_thoi_gian["bat_dau"]} đến {khoang_thoi_gian["ket_thuc"]}')
        
        phan_bo_loai = tom_tat.get('phan_bo_loai_giao_dich', {})
        if phan_bo_loai:
            self.stdout.write('\n📈 PHÂN BỐ LOẠI GIAO DỊCH:')
            for loai, so_luong in phan_bo_loai.items():
                self.stdout.write(f'   • {loai}: {so_luong:,} giao dịch')
        
        so_nguoi_dung = tom_tat.get('so_nguoi_dung_duy_nhat', {})
        self.stdout.write(f'\n👥 Số người gửi duy nhất: {so_nguoi_dung.get("nguoi_gui", 0):,}')
        self.stdout.write(f'👤 Số người nhận duy nhất: {so_nguoi_dung.get("nguoi_nhan", 0):,}')
        
        self.stdout.write('\n' + '='*50)
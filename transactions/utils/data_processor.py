"""
Tiện ích xử lý và làm sạch dữ liệu giao dịch ngân hàng
Hỗ trợ xử lý file Excel và chuẩn bị dữ liệu cho machine learning
"""
import pandas as pd
import numpy as np
from datetime import datetime
import re
from typing import Dict, List, Tuple, Optional

class BoXuLyDuLieu:
    """
    Lớp xử lý và làm sạch dữ liệu giao dịch ngân hàng
    """
    
    def __init__(self):
        # Các cột bắt buộc trong dữ liệu
        self.cac_cot_bat_buoc = [
            'ma_giao_dich', 'id_nguoi_gui', 'id_nguoi_nhan', 'so_tien',
            'so_du_tai_khoan', 'ngay_gio_giao_dich', 'loai_giao_dich',
            'trang_thai_giao_dich', 'noi_dung_giao_dich', 'co_gian_lan'
        ]
        
        # Bảng ánh xạ tên cột từ tiếng Việt/Anh sang định dạng chuẩn
        self.bang_anh_xa_cot = {
            # Tiếng Việt có dấu (thường)
            'mã giao dịch': 'ma_giao_dich',
            'id người gửi': 'id_nguoi_gui', 
            'id người nhận': 'id_nguoi_nhan',
            'số tiền giao dịch': 'so_tien',
            'số dư tài khoản': 'so_du_tai_khoan',
            'ngày giờ chuyển': 'ngay_gio_giao_dich',
            'loại giao dịch': 'loai_giao_dich',
            'trạng thái giao dịch': 'trang_thai_giao_dich',
            'nội dung giao dịch': 'noi_dung_giao_dich',
            'cờ gian lận': 'co_gian_lan',
            # Tiếng Việt có dấu (với underscore - sau khi normalize)
            'mã_giao_dịch': 'ma_giao_dich',
            'id_người_gửi': 'id_nguoi_gui', 
            'id_người_nhận': 'id_nguoi_nhan',
            'số_tiền_giao_dịch': 'so_tien',
            'số_dư_tài_khoản': 'so_du_tai_khoan',
            'ngày_giờ_chuyển': 'ngay_gio_giao_dich',
            'loại_giao_dịch': 'loai_giao_dich',
            'trạng_thái_giao_dịch': 'trang_thai_giao_dich',
            'nội_dung_giao_dịch': 'noi_dung_giao_dich',
            'cờ_gian_lận': 'co_gian_lan',
            # Tiếng Việt không dấu
            'ma_giao_dich': 'ma_giao_dich',
            'id_nguoi_gui': 'id_nguoi_gui',
            'id_nguoi_nhan': 'id_nguoi_nhan', 
            'so_tien_giao_dich': 'so_tien',
            'so_du_tai_khoan': 'so_du_tai_khoan',
            'ngay_gio_chuyen': 'ngay_gio_giao_dich',
            'loai_giao_dich': 'loai_giao_dich',
            'trang_thai_giao_dich': 'trang_thai_giao_dich',
            'noi_dung_giao_dich': 'noi_dung_giao_dich',
            'co_gian_lan': 'co_gian_lan',
            # Tiếng Anh
            'transaction_id': 'ma_giao_dich',
            'sender_id': 'id_nguoi_gui',
            'receiver_id': 'id_nguoi_nhan',
            'amount': 'so_tien',
            'account_balance': 'so_du_tai_khoan',
            'transaction_date': 'ngay_gio_giao_dich',
            'transaction_type': 'loai_giao_dich',
            'transaction_status': 'trang_thai_giao_dich',
            'description': 'noi_dung_giao_dich',
            'is_fraud': 'co_gian_lan'
        }
        
        # Ánh xạ loại giao dịch
        self.anh_xa_loai_giao_dich = {
            'transfer': 'CHUYEN_KHOAN',
            'chuyển khoản': 'CHUYEN_KHOAN',
            'chuyen khoan': 'CHUYEN_KHOAN',
            'payment': 'THANH_TOAN',
            'thanh toán': 'THANH_TOAN',
            'thanh toan': 'THANH_TOAN',
            'withdrawal': 'RUT_TIEN',
            'rút tiền': 'RUT_TIEN',
            'rut tien': 'RUT_TIEN',
            'deposit': 'GUI_TIEN',
            'gửi tiền': 'GUI_TIEN',
            'gui tien': 'GUI_TIEN',
            'purchase': 'MUA_SAM',
            'mua sắm': 'MUA_SAM',
            'mua sam': 'MUA_SAM',
            'other': 'KHAC',
            'khác': 'KHAC',
            'khac': 'KHAC'
        }
    
    def doc_file_excel(self, duong_dan_file: str) -> pd.DataFrame:
        """
        Đọc file Excel và trả về DataFrame
        
        Args:
            duong_dan_file: Đường dẫn đến file Excel
            
        Returns:
            DataFrame chứa dữ liệu từ file Excel
        """
        try:
            df = pd.read_excel(duong_dan_file)
            print(f"Đã đọc thành công file Excel với {len(df)} dòng và {len(df.columns)} cột")
            return df
        except Exception as e:
            raise ValueError(f"Lỗi khi đọc file Excel: {str(e)}")
    
    def chuan_hoa_ten_cot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa tên cột theo bảng ánh xạ
        
        Args:
            df: DataFrame cần chuẩn hóa
            
        Returns:
            DataFrame với tên cột đã được chuẩn hóa
        """
        df_sao_chep = df.copy()
        
        # Chuyển tên cột về chữ thường và loại bỏ khoảng trắng thừa
        df_sao_chep.columns = df_sao_chep.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Áp dụng bảng ánh xạ
        ten_cot_moi = {}
        for cot_cu in df_sao_chep.columns:
            if cot_cu in self.bang_anh_xa_cot:
                ten_cot_moi[cot_cu] = self.bang_anh_xa_cot[cot_cu]
            else:
                ten_cot_moi[cot_cu] = cot_cu
        
        df_sao_chep.rename(columns=ten_cot_moi, inplace=True)
        
        print(f"Đã chuẩn hóa tên cột: {list(df_sao_chep.columns)}")
        return df_sao_chep
    
    def lam_sach_du_lieu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch dữ liệu giao dịch
        
        Args:
            df: DataFrame cần làm sạch
            
        Returns:
            DataFrame đã được làm sạch
        """
        df_sach = df.copy()
        
        print("Bắt đầu quá trình làm sạch dữ liệu...")
        
        # Xóa các dòng trùng lặp
        so_dong_truoc = len(df_sach)
        df_sach = df_sach.drop_duplicates(subset=['ma_giao_dich'], keep='first')
        so_dong_bi_xoa = so_dong_truoc - len(df_sach)
        if so_dong_bi_xoa > 0:
            print(f"Đã xóa {so_dong_bi_xoa} dòng trùng lặp")
        
        # Xử lý giá trị thiếu
        df_sach = self._xu_ly_gia_tri_thieu(df_sach)
        
        # Chuẩn hóa kiểu dữ liệu
        df_sach = self._chuan_hoa_kieu_du_lieu(df_sach)
        
        # Làm sạch các trường text
        df_sach = self._lam_sach_truong_text(df_sach)
        
        # Xác thực dữ liệu
        df_sach = self._xac_thuc_du_lieu(df_sach)
        
        print(f"Hoàn thành làm sạch dữ liệu. Còn lại {len(df_sach)} dòng hợp lệ")
        return df_sach
    
    def _xu_ly_gia_tri_thieu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xử lý các giá trị thiếu trong dữ liệu
        """
        df_xu_ly = df.copy()
        
        # Điền giá trị mặc định cho các cột không quan trọng
        if 'noi_dung_giao_dich' in df_xu_ly.columns:
            df_xu_ly['noi_dung_giao_dich'].fillna('Không có mô tả', inplace=True)
        
        if 'trang_thai_giao_dich' in df_xu_ly.columns:
            df_xu_ly['trang_thai_giao_dich'].fillna('HOAN_THANH', inplace=True)
        
        if 'co_gian_lan' in df_xu_ly.columns:
            df_xu_ly['co_gian_lan'].fillna(False, inplace=True)
        
        # Xóa các dòng có giá trị thiếu trong cột quan trọng
        cac_cot_quan_trong = ['ma_giao_dich', 'so_tien', 'ngay_gio_giao_dich']
        so_dong_truoc = len(df_xu_ly)
        df_xu_ly = df_xu_ly.dropna(subset=cac_cot_quan_trong)
        so_dong_bi_xoa = so_dong_truoc - len(df_xu_ly)
        if so_dong_bi_xoa > 0:
            print(f"Đã xóa {so_dong_bi_xoa} dòng có giá trị thiếu trong cột quan trọng")
        
        return df_xu_ly
    
    def _chuan_hoa_kieu_du_lieu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa kiểu dữ liệu cho các cột
        """
        df_chuan_hoa = df.copy()
        
        # Chuyển đổi kiểu số
        if 'so_tien' in df_chuan_hoa.columns:
            df_chuan_hoa['so_tien'] = pd.to_numeric(df_chuan_hoa['so_tien'], errors='coerce')
        
        if 'so_du_tai_khoan' in df_chuan_hoa.columns:
            df_chuan_hoa['so_du_tai_khoan'] = pd.to_numeric(df_chuan_hoa['so_du_tai_khoan'], errors='coerce')
        
        # Chuyển đổi kiểu ngày tháng
        if 'ngay_gio_giao_dich' in df_chuan_hoa.columns:
            df_chuan_hoa['ngay_gio_giao_dich'] = pd.to_datetime(df_chuan_hoa['ngay_gio_giao_dich'], errors='coerce')
        
        # Chuyển đổi kiểu boolean
        if 'co_gian_lan' in df_chuan_hoa.columns:
            df_chuan_hoa['co_gian_lan'] = df_chuan_hoa['co_gian_lan'].astype(bool)
        
        return df_chuan_hoa
    
    def _lam_sach_truong_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Làm sạch các trường text
        """
        df_sach = df.copy()
        
        cac_cot_text = ['noi_dung_giao_dich', 'loai_giao_dich', 'trang_thai_giao_dich']
        
        for cot in cac_cot_text:
            if cot in df_sach.columns:
                # Loại bỏ khoảng trắng thừa và chuẩn hóa
                df_sach[cot] = df_sach[cot].astype(str).str.strip()
                df_sach[cot] = df_sach[cot].str.replace(r'[^\w\s]', '', regex=True)
        
        # Chuẩn hóa loại giao dịch
        if 'loai_giao_dich' in df_sach.columns:
            df_sach['loai_giao_dich'] = df_sach['loai_giao_dich'].str.lower()
            df_sach['loai_giao_dich'] = df_sach['loai_giao_dich'].map(self.anh_xa_loai_giao_dich).fillna('KHAC')
        
        return df_sach
    
    def _xac_thuc_du_lieu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Xác thực tính hợp lệ của dữ liệu
        """
        df_hop_le = df.copy()
        
        # Lọc ra các giao dịch có số tiền âm
        if 'so_tien' in df_hop_le.columns:
            so_dong_truoc = len(df_hop_le)
            df_hop_le = df_hop_le[df_hop_le['so_tien'] >= 0]
            so_dong_bi_xoa = so_dong_truoc - len(df_hop_le)
            if so_dong_bi_xoa > 0:
                print(f"Đã xóa {so_dong_bi_xoa} giao dịch có số tiền âm")
        
        # Lọc ra các giao dịch có ngày không hợp lệ
        if 'ngay_gio_giao_dich' in df_hop_le.columns:
            so_dong_truoc = len(df_hop_le)
            df_hop_le = df_hop_le.dropna(subset=['ngay_gio_giao_dich'])
            so_dong_bi_xoa = so_dong_truoc - len(df_hop_le)
            if so_dong_bi_xoa > 0:
                print(f"Đã xóa {so_dong_bi_xoa} giao dịch có ngày không hợp lệ")
        
        return df_hop_le
    
    def trich_xuat_dac_trung(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trích xuất đặc trưng cho machine learning
        
        Args:
            df: DataFrame gốc
            
        Returns:
            DataFrame với các đặc trưng đã được trích xuất
        """
        df_dac_trung = df.copy()
        
        print("Bắt đầu trích xuất đặc trưng...")
        
        if 'ngay_gio_giao_dich' in df_dac_trung.columns:
            # Trích xuất đặc trưng thời gian
            df_dac_trung['gio_trong_ngay'] = df_dac_trung['ngay_gio_giao_dich'].dt.hour
            df_dac_trung['thu_trong_tuan'] = df_dac_trung['ngay_gio_giao_dich'].dt.dayofweek
            df_dac_trung['thang_trong_nam'] = df_dac_trung['ngay_gio_giao_dich'].dt.month
            df_dac_trung['la_cuoi_tuan'] = df_dac_trung['thu_trong_tuan'].isin([5, 6])
            df_dac_trung['la_gio_cao_diem'] = df_dac_trung['gio_trong_ngay'].isin(range(9, 17))
        
        if 'so_tien' in df_dac_trung.columns:
            # Trích xuất đặc trưng số tiền
            df_dac_trung['log_so_tien'] = np.log1p(df_dac_trung['so_tien'])
            df_dac_trung['la_so_tien_cao'] = df_dac_trung['so_tien'] > df_dac_trung['so_tien'].quantile(0.95)
            df_dac_trung['la_so_tien_thap'] = df_dac_trung['so_tien'] < df_dac_trung['so_tien'].quantile(0.05)
        
        # Mã hóa các đặc trưng phân loại
        df_dac_trung = self._ma_hoa_dac_trung_phan_loai(df_dac_trung)
        
        print(f"Đã trích xuất {len(df_dac_trung.columns)} đặc trưng")
        return df_dac_trung
    
    def _ma_hoa_dac_trung_phan_loai(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mã hóa các đặc trưng phân loại
        """
        df_ma_hoa = df.copy()
        
        cac_cot_phan_loai = ['loai_giao_dich', 'trang_thai_giao_dich']
        
        for cot in cac_cot_phan_loai:
            if cot in df_ma_hoa.columns:
                # Label encoding đơn giản
                cac_gia_tri_duy_nhat = df_ma_hoa[cot].unique()
                bang_ma_hoa = {gia_tri: chi_so for chi_so, gia_tri in enumerate(cac_gia_tri_duy_nhat)}
                df_ma_hoa[f'{cot}_ma_hoa'] = df_ma_hoa[cot].map(bang_ma_hoa)
        
        return df_ma_hoa
    
    def tao_tom_tat_du_lieu(self, df: pd.DataFrame) -> Dict:
        """
        Tạo báo cáo tóm tắt dữ liệu
        
        Args:
            df: DataFrame cần tóm tắt
            
        Returns:
            Dictionary chứa thông tin tóm tắt
        """
        tom_tat = {
            'tong_so_giao_dich': len(df),
            'khoang_thoi_gian': {
                'bat_dau': df['ngay_gio_giao_dich'].min() if 'ngay_gio_giao_dich' in df.columns else None,
                'ket_thuc': df['ngay_gio_giao_dich'].max() if 'ngay_gio_giao_dich' in df.columns else None
            },
            'thong_ke_so_tien': {
                'tong_cong': float(df['so_tien'].sum()) if 'so_tien' in df.columns else 0,
                'trung_binh': float(df['so_tien'].mean()) if 'so_tien' in df.columns else 0,
                'trung_vi': float(df['so_tien'].median()) if 'so_tien' in df.columns else 0,
                'do_lech_chuan': float(df['so_tien'].std()) if 'so_tien' in df.columns else 0,
                'gia_tri_nho_nhat': float(df['so_tien'].min()) if 'so_tien' in df.columns else 0,
                'gia_tri_lon_nhat': float(df['so_tien'].max()) if 'so_tien' in df.columns else 0
            },
            'thong_ke_gian_lan': {
                'tong_so_gian_lan': int(df['co_gian_lan'].sum()) if 'co_gian_lan' in df.columns else 0,
                'ty_le_gian_lan': float(df['co_gian_lan'].mean()) if 'co_gian_lan' in df.columns else 0
            },
            'phan_bo_loai_giao_dich': df['loai_giao_dich'].value_counts().to_dict() if 'loai_giao_dich' in df.columns else {},
            'gia_tri_thieu': df.isnull().sum().to_dict(),
            'so_nguoi_dung_duy_nhat': {
                'nguoi_gui': df['id_nguoi_gui'].nunique() if 'id_nguoi_gui' in df.columns else 0,
                'nguoi_nhan': df['id_nguoi_nhan'].nunique() if 'id_nguoi_nhan' in df.columns else 0
            }
        }
        
        return tom_tat
    
    def luu_du_lieu_da_xu_ly(self, df: pd.DataFrame, duong_dan_file: str) -> None:
        """
        Lưu dữ liệu đã xử lý ra file
        
        Args:
            df: DataFrame cần lưu
            duong_dan_file: Đường dẫn file đích
        """
        try:
            if duong_dan_file.endswith('.xlsx'):
                df.to_excel(duong_dan_file, index=False)
            elif duong_dan_file.endswith('.csv'):
                df.to_csv(duong_dan_file, index=False, encoding='utf-8')
            else:
                raise ValueError("Chỉ hỗ trợ định dạng .xlsx và .csv")
            
            print(f"Đã lưu dữ liệu thành công vào {duong_dan_file}")
        except Exception as e:
            raise ValueError(f"Lỗi khi lưu file: {str(e)}")
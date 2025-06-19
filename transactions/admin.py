from django.contrib import admin
from .models import Transaction, CumGiaoDich, DuDoanGianLan

@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = ['ma_giao_dich', 'id_nguoi_gui', 'id_nguoi_nhan', 'so_tien', 'loai_giao_dich', 'ngay_gio_giao_dich', 'co_gian_lan']
    list_filter = ['loai_giao_dich', 'trang_thai_giao_dich', 'co_gian_lan', 'ngay_gio_giao_dich']
    search_fields = ['ma_giao_dich', 'id_nguoi_gui', 'id_nguoi_nhan', 'noi_dung_giao_dich']
    readonly_fields = ['ngay_tao', 'ngay_cap_nhat']
    date_hierarchy = 'ngay_gio_giao_dich'
    
    fieldsets = (
        ('Thông tin cơ bản', {
            'fields': ('ma_giao_dich', 'id_nguoi_gui', 'id_nguoi_nhan')
        }),
        ('Chi tiết giao dịch', {
            'fields': ('so_tien', 'so_du_tai_khoan', 'ngay_gio_giao_dich', 'loai_giao_dich', 'trang_thai_giao_dich')
        }),
        ('Nội dung và bảo mật', {
            'fields': ('noi_dung_giao_dich', 'co_gian_lan')
        }),
        ('Thông tin hệ thống', {
            'fields': ('ngay_tao', 'ngay_cap_nhat'),
            'classes': ('collapse',)
        }),
    )

@admin.register(CumGiaoDich)
class CumGiaoDichAdmin(admin.ModelAdmin):
    list_display = ['giao_dich', 'id_cum', 'nhan_cum', 'ngay_tao']
    list_filter = ['id_cum', 'nhan_cum']
    search_fields = ['giao_dich__ma_giao_dich', 'nhan_cum']
    readonly_fields = ['ngay_tao']

@admin.register(DuDoanGianLan)
class DuDoanGianLanAdmin(admin.ModelAdmin):
    list_display = ['giao_dich', 'diem_gian_lan', 'du_doan_gian_lan', 'phien_ban_mo_hinh', 'ngay_du_doan']
    list_filter = ['du_doan_gian_lan', 'phien_ban_mo_hinh', 'ngay_du_doan']
    search_fields = ['giao_dich__ma_giao_dich']
    readonly_fields = ['ngay_du_doan']
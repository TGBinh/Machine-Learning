# 🏦 Hệ thống Phân tích Tài chính Cá nhân - PHIÊN BẢN CẬP NHẬT

## 📋 Tổng quan
Hệ thống được cập nhật với **2 chức năng riêng biệt**:

### 🔍 1. Phát hiện Gian lận (Fraud Detection)
- **Mục đích**: Phân tích dữ liệu giao dịch của nhiều người dùng để phát hiện giao dịch bất thường
- **Thuật toán**: K-Means Clustering với Elbow Method
- **Dữ liệu**: `phgl.xlsx` (2512 giao dịch, 16 cột)
- **Tính năng**:
  - Tự động xác định số cụm tối ưu
  - Phát hiện giao dịch nghi ngờ
  - Biểu đồ trực quan (Elbow chart, Scatter plot)
  - Báo cáo chi tiết các giao dịch bất thường

### 💰 2. Phân tích Tài chính Cá nhân (Personal Finance)
- **Mục đích**: Phân tích chi tiêu cá nhân và đưa ra gợi ý tiết kiệm
- **Phương pháp**: Phân tích thống kê và phân loại chi tiêu
- **Dữ liệu**: `canhan.xlsx` (1949 giao dịch, 8 cột)
- **Tính năng**:
  - Phân loại chi tiêu theo danh mục
  - Gợi ý tiết kiệm thông minh
  - Báo cáo xu hướng chi tiêu
  - Phân tích chi tiêu theo thời gian

## 🚀 Cách sử dụng

### Lần đầu cài đặt:
1. Chạy `setup_and_run.bat`
2. Đợi quá trình cài đặt hoàn tất
3. Hệ thống sẽ tự động khởi động

### Lần sau:
1. Chạy `run_server.bat`
2. Mở trình duyệt: `http://127.0.0.1:8000/`

### Sử dụng tính năng:
1. **Trang chủ**: Chọn một trong hai tùy chọn
2. **Phát hiện gian lận**: 
   - Click "Phát hiện Gian lận"
   - Upload file `phgl.xlsx`
   - Xem kết quả phân tích
3. **Phân tích cá nhân**:
   - Click "Phân tích Tài chính Cá nhân"  
   - Upload file `canhan.xlsx`
   - Xem báo cáo và gợi ý

## 📂 Cấu trúc file dữ liệu

### phgl.xlsx (Fraud Detection)
```
TransactionID, AccountID, TransactionAmount, TransactionDate, 
TransactionType, Location, DeviceID, IP Address, MerchantID, 
Channel, CustomerAge, CustomerOccupation, TransactionDuration, 
LoginAttempts, AccountBalance, PreviousTransactionDate
```

### canhan.xlsx (Personal Finance)
```
Mã giao dịch, Thời gian, ID người nhận, Số tiền, 
Số dư hiện tại, Trạng thái, Loại giao dịch, Nội dung giao dịch
```

## 🔧 Yêu cầu kỹ thuật
- Python 3.8+
- Django 4.0+
- Libraries: pandas, scikit-learn, matplotlib, seaborn, openpyxl

## 📈 Kết quả kiểm thử
```
✅ Fraud Detection: PASSED
   - Tải 2512 giao dịch thành công
   - K-means tự động tìm k=3
   - Phát hiện 5% giao dịch nghi ngờ

✅ Personal Finance: PASSED  
   - Tải 1949 giao dịch thành công
   - Phân tích 11.68M VND chi tiêu
   - Phân loại thành 2 danh mục
   - Tạo gợi ý tiết kiệm
```

## 🎯 Tính năng chính

### Phát hiện Gian lận:
- ✅ K-Means Clustering với Elbow Method
- ✅ Tự động xác định số cụm tối ưu
- ✅ Phát hiện outliers/anomalies
- ✅ Biểu đồ trực quan kết quả
- ✅ Báo cáo giao dịch nghi ngờ

### Phân tích Cá nhân:
- ✅ Phân loại chi tiêu thông minh
- ✅ Gợi ý tiết kiệm cá nhân hóa
- ✅ Phân tích xu hướng chi tiêu
- ✅ Báo cáo chi tiết và biểu đồ
- ✅ Xuất báo cáo PDF (sắp có)

## 🔒 Bảo mật
- File được xóa tự động sau khi xử lý
- Không lưu trữ dữ liệu cá nhân
- Xử lý tạm thời trong session
- Tuân thủ quy định bảo vệ dữ liệu

## 📞 Hỗ trợ
Nếu gặp vấn đề, kiểm tra:
1. File dữ liệu đúng định dạng (.xlsx)
2. Kết nối internet ổn định
3. Python và dependencies đã cài đặt
4. Chạy lại `setup_and_run.bat` nếu cần

---
*Phiên bản cập nhật: 2.0 - Hỗ trợ phân tích gian lận và tài chính cá nhân riêng biệt*
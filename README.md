

###  Đồ án cuối kỳ - Đại học Sư phạm Kỹ thuật TP.HCM

---

##  Giới thiệu

Dự án này nhằm **phân tích và dự báo giá trị "VN30_Lần cuối"** trong bộ dữ liệu PCA, thông qua việc kết hợp **các mô hình hồi quy truyền thống, Machine Learning, Deep Learning và mô hình chuỗi thời gian ARDL**.  
Mục tiêu là xác định **mô hình dự báo hiệu quả nhất** dựa trên các chỉ tiêu RMSE, MAE, và MAPE, đồng thời trực quan hóa kết quả để hỗ trợ đánh giá và ra quyết định.

---

##  Công nghệ & Thư viện sử dụng

- **Ngôn ngữ chính:** Python 3.x  
- **Xử lý & trực quan dữ liệu:** `pandas`, `numpy`, `matplotlib`, `seaborn`  
- **Machine Learning:** `scikit-learn`, `xgboost`  
- **Deep Learning:** `tensorflow`, `keras`  
- **Kinh tế lượng & chuỗi thời gian:** `statsmodels` (ARDL)  
- **Chuẩn hóa dữ liệu:** `StandardScaler`

---

## 🧩 Mô tả các bước chính

1. **Tiền xử lý dữ liệu**
   - Đọc dữ liệu từ file `PCA.xlsx`
   - Chuyển định dạng thời gian (`Ngày`)
   - Chia tập huấn luyện & kiểm tra theo mốc thời gian

2. **Xây dựng & huấn luyện các mô hình**
   - `Linear Regression`
   - `Lasso Regression`
   - `ARDL` (Autoregressive Distributed Lag)
   - `Random Forest`
   - `XGBoost`
   - `LSTM`
   - `GRU`

3. **Đánh giá & so sánh mô hình**
   - Sử dụng các chỉ số:
     - RMSE (Root Mean Squared Error)
     - MAE (Mean Absolute Error)
     - MAPE (Mean Absolute Percentage Error)
   - Xác định mô hình có sai số thấp nhất.

4. **Trực quan hóa kết quả**
   - Biểu đồ đường: So sánh giá trị thực tế và dự báo của tất cả mô hình  
   - Biểu đồ cột: So sánh sai số (RMSE)  
   - Biểu đồ riêng cho mô hình tốt nhất

---

##  Kết quả chính

- So sánh toàn bộ mô hình trong cùng khung dữ liệu test
- Tự động xác định mô hình có hiệu suất cao nhất  
- Trực quan hóa giúp phân tích xu hướng và sai số dễ dàng hơn  

**Ví dụ đầu ra:**
- Bảng tổng hợp RMSE/MAE/MAPE
- Biểu đồ đường VN30_Lần cuối (thực tế vs dự báo)
- Biểu đồ cột RMSE giữa các mô hình
- Biểu đồ riêng mô hình tốt nhất

---

##  Mục tiêu học thuật

- Vận dụng kiến thức **phân tích dữ liệu, kinh tế lượng, machine learning và deep learning** vào bài toán thực tế.  
- Thực hành quy trình phân tích đầy đủ:
  - Thu thập & tiền xử lý dữ liệu
  - Lựa chọn mô hình
  - Đánh giá & trình bày kết quả trực quan

---
 VN30-Predict-Project
│
├── PCA.xlsx                # Dữ liệu gốc
├── vn30_prediction.py      # Script chính (gồm toàn bộ mô hình)
└── README.md               # File mô tả dự án


## 🛠️ Cách chạy dự án

### 1️⃣ Cài đặt thư viện
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow xgboost statsmodels openpyxl

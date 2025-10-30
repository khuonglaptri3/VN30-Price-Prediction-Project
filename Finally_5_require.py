# =====================================================
#  YÊU CẦU 5 — THỰC HIỆN PCA TRÊN DATA2
# (Chọn số thành phần chính sao cho % phương sai tích lũy >= 70%)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === 1. Đọc dữ liệu ===
path_data = r"F:\VN30-Price-Prediction-Project\data2.xlsx"   # dùng data2 có cả cột Ngày và VN30_Lần cuối
df = pd.read_excel(path_data)
print(f" Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

# === 2. Xác định cột ngày & cột VN30 mục tiêu ===
date_col = None
vn30_col = None
for col in df.columns:
    if "Ngày" in col or "Date" in col:
        date_col = col
    if "VN30" in col and "Lần cuối" in col:
        vn30_col = col

if date_col is None:
    print(" Không tìm thấy cột ngày — sẽ bỏ qua.")
if vn30_col is None:
    raise ValueError(" Không tìm thấy cột VN30_Lần cuối trong dữ liệu.")

# === 3. Giữ lại các biến số (loại bỏ cột ngày, ký tự) ===
data = df.select_dtypes(include=[np.number]).dropna()
print(f" Dữ liệu số dùng cho PCA: {data.shape[0]} dòng, {data.shape[1]} biến")

# === 4. Chuẩn hóa dữ liệu ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# === 5. PCA ban đầu để tính % phương sai tích lũy ===
pca_temp = PCA()
pca_temp.fit(scaled_data)
cum_var = np.cumsum(pca_temp.explained_variance_ratio_) * 100
n_components = np.argmax(cum_var >= 70) + 1
print(f" Số thành phần chính cần chọn: {n_components} (đạt {cum_var[n_components-1]:.2f}% phương sai tích lũy)")

# === 6. PCA chính thức ===
pca = PCA(n_components=n_components)
pca_data = pca.fit_transform(scaled_data)

# === 7. Tạo DataFrame kết quả PCA ===
pca_df = pd.DataFrame(
    pca_data,
    columns=[f'PC{i+1}' for i in range(n_components)],
    index=data.index
)

# === 8. Thêm cột ngày & VN30_Lần cuối (nếu có) ===
if date_col in df.columns:
    pca_df.insert(0, date_col, df.loc[pca_df.index, date_col].values)
pca_df[vn30_col] = df.loc[pca_df.index, vn30_col].values

# === 9. Lưu kết quả ra Excel ===
out_path = r"F:\VN30-Price-Prediction-Project\PCA.xlsx"
pca_df.to_excel(out_path, index=False)
print(f" Đã lưu kết quả PCA tại: {out_path}")
print(f"   File gồm {pca_df.shape[1]} cột (bao gồm {n_components} thành phần chính + Ngày + VN30_Lần cuối).")

# === 10. Biểu đồ phương sai tích lũy ===
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', linestyle='--', color='teal')
plt.axhline(y=70, color='r', linestyle='--', label='Ngưỡng 70%')
plt.axvline(x=n_components, color='orange', linestyle='--', label=f'{n_components} thành phần')
plt.title(' Biểu đồ phương sai tích lũy của các thành phần chính (PCA)')
plt.xlabel('Số thành phần chính')
plt.ylabel('Tỷ lệ phương sai tích lũy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 11. Biểu đồ tỷ lệ phương sai từng thành phần ===
plt.figure(figsize=(8, 4))
plt.bar(range(1, n_components + 1),
        pca.explained_variance_ratio_[:n_components] * 100,
        color='skyblue', edgecolor='black')
plt.title(' Tỷ lệ phương sai giải thích bởi từng thành phần chính')
plt.xlabel('Thành phần chính (PC)')
plt.ylabel('Tỷ lệ phương sai (%)')
plt.tight_layout()
plt.show()

# === 12. Diễn giải kết quả ===
print("\n DIỄN GIẢI KẾT QUẢ PCA:")
print(f"   - Tổng số biến gốc: {data.shape[1]}")
print(f"   - Sau PCA, chỉ cần {n_components} thành phần đầu tiên là đủ giải thích {cum_var[n_components-1]:.2f}% biến thiên của dữ liệu.")
print("   - Giảm chiều giúp loại bỏ nhiễu, giữ lại đặc trưng chính trong biến động cổ phiếu.")
print("   - File PCA.xlsx đã kèm cả cột thời gian và VN30_Lần cuối để tiện cho mô hình hóa tiếp theo.")

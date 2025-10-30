import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_excel(r'F:\KL\KQ_VN30.xlsx')

#  Điền khuyết thiếu cho cột số (bằng giá trị trung bình)
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

#  Điền khuyết thiếu cho cột chữ (bằng giá trị xuất hiện nhiều nhất)
for col in df.select_dtypes(exclude=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Kiểm tra lại
print(df.isnull().sum())

# Lưu file mới
df.to_excel('data1.xlsx', index=False)
print(" Đã lưu file data1.xlsx thành công.")

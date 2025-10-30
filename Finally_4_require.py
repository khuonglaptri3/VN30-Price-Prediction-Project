# =====================================================
# 🧠 VN30 INSIGHT ANALYZER — HIỂN THỊ ĐẦY ĐỦ + GỘP THEO NGÂN HÀNG
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")

# === 1. Đọc dữ liệu gốc ===
path = r"F:\VN30-Price-Prediction-Project\data1.xlsx"
df = pd.read_excel(path)
print(f"✅ Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

# === 2. Lọc cột số & tìm cột VN30 tiêu biểu ===
numeric_df = df.select_dtypes(include=[np.number]).copy()
vn30_col = None
for col in numeric_df.columns:
    if "VN30" in col and ("Lần cuối" in col or "Mở" in col or "Cao" in col or "Thấp" in col):
        vn30_col = col
        break
if vn30_col is None:
    raise ValueError("❌ Không tìm thấy cột VN30 phù hợp trong dữ liệu.")
print(f" Phân tích theo cột: {vn30_col}")

# === 3. Tính tương quan với VN30 và lọc theo ngưỡng ===
corr_matrix = numeric_df.corr(method="pearson")
corr_with_vn30 = corr_matrix[vn30_col].sort_values(ascending=False)
filtered_corr = corr_with_vn30[abs(corr_with_vn30) >= 0.1]

print(f"\n Các biến có |r| ≥ 0.1 với VN30: {len(filtered_corr)} biến")
print(filtered_corr)
print("-" * 60)

# === 4. BIỂU ĐỒ 1: HIỂN THỊ ĐẦY ĐỦ TẤT CẢ CÁC BIẾN ===
filtered_corr_df = filtered_corr.reset_index()
filtered_corr_df.columns = ["Biến", "Hệ_số_tương_quan"]

# Chia 2 nhóm: dương & âm
pos_corr = filtered_corr_df[filtered_corr_df["Hệ_số_tương_quan"] > 0].sort_values(by="Hệ_số_tương_quan", ascending=True)
neg_corr = filtered_corr_df[filtered_corr_df["Hệ_số_tương_quan"] < 0].sort_values(by="Hệ_số_tương_quan", ascending=True)

# KHÔNG giới hạn top_n - lấy tất cả
combined = pd.concat([neg_corr, pos_corr])
combined["Nhóm"] = np.where(combined["Hệ_số_tương_quan"] > 0, "Tương quan DƯƠNG", "Tương quan ÂM")

# Biểu đồ Plotly (chia 2 phía) - HIỂN THỊ FULL
fig1 = px.bar(
    combined,
    x="Hệ_số_tương_quan",
    y="Biến",
    orientation="h",
    color="Nhóm",
    color_discrete_map={"Tương quan DƯƠNG": "#2ca02c", "Tương quan ÂM": "#d62728"},
    title=f" Biểu đồ tương quan  giữa các biến & {vn30_col}",
    text="Hệ_số_tương_quan"
)

fig1.add_vline(x=0, line_dash="dash", line_color="gray")
fig1.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig1.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=max(800, len(combined) * 20), 
    xaxis_title="Hệ số tương quan (r)",
    yaxis_title="Biến cổ phiếu",
)
fig1.show()

# === 5. BIỂU ĐỒ 2: GỘP THEO NGÂN HÀNG/MÃ CỔ PHIẾU ===
# Trích xuất mã cổ phiếu từ tên cột (VD: "ACB_Lần cuối" -> "ACB")
def extract_symbol(col_name):
    """Trích xuất mã cổ phiếu từ tên cột."""
    if "_" in col_name:
        return col_name.split("_")[0]
    return col_name

combined["Mã_CP"] = combined["Biến"].apply(extract_symbol)

# Tính trung bình tương quan của từng mã cổ phiếu
symbol_corr = combined.groupby("Mã_CP")["Hệ_số_tương_quan"].mean().reset_index()
symbol_corr.columns = ["Mã_CP", "Tương_quan_TB"]
symbol_corr = symbol_corr.sort_values(by="Tương_quan_TB", ascending=True)

# Phân loại dương/âm
symbol_corr["Nhóm"] = np.where(symbol_corr["Tương_quan_TB"] > 0, "Tương quan DƯƠNG", "Tương quan ÂM")

# Biểu đồ gộp theo mã cổ phiếu
fig2 = px.bar(
    symbol_corr,
    x="Tương_quan_TB",
    y="Mã_CP",
    orientation="h",
    color="Nhóm",
    color_discrete_map={"Tương quan DƯƠNG": "#2ca02c", "Tương quan ÂM": "#d62728"},
    title=" Tương quan của các biến với VN30 ",
    text="Tương_quan_TB"
)

fig2.add_vline(x=0, line_dash="dash", line_color="gray")
fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig2.update_layout(
    yaxis={"categoryorder": "total ascending"},
    height=max(600, len(symbol_corr) * 25),
    xaxis_title="Hệ số tương quan trung bình (r)",
    yaxis_title="Mã cổ phiếu",
)
fig2.show()

# === 8. Lưu dữ liệu các biến có tương quan |r| ≥ 0.1 ===
selected_cols = list(set(top_features + [vn30_col]))
data_filtered = numeric_df[selected_cols].copy()

# Giữ thêm cột Ngày / Date nếu có trong file gốc
date_cols = [c for c in df.columns if c.lower() in ["ngày", "date", "time", "datetime"]]
if date_cols:
    data_filtered = pd.concat([df[date_cols], data_filtered], axis=1)
    print(f" Đã thêm cột thời gian: {date_cols}")
else:
    print(" Không tìm thấy cột 'Ngày' hoặc 'Date' trong dữ liệu.")

out_path = r"F:\VN30-Price-Prediction-Project\data2.xlsx"
data_filtered.to_excel(out_path, index=False)
print(f"\n Đã lưu dữ liệu các biến có |r| ≥ 0.1 vào: {out_path}")
print(f" Số biến được giữ lại: {len(selected_cols)} (kèm {len(date_cols)} cột thời gian nếu có)")

# === 9. Chuẩn bị dữ liệu cho PCA & t-SNE ===
numeric_clean = data_filtered.dropna(axis=0, how='any').copy()
numeric_clean = numeric_clean.select_dtypes(include=[np.number])
print(f" Dữ liệu sau khi loại bỏ cột thời gian: {numeric_clean.shape}")

if numeric_clean.shape[0] == 0:
    raise ValueError("❌ Dữ liệu sau dropna rỗng. Hãy xử lý NaN trước.")

scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_clean)

# === PCA (2 thành phần) ===
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(scaled_data)

# === t-SNE ===
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init='pca',
    learning_rate='auto',
    method='barnes_hut',
    max_iter=1000
)
tsne_result = tsne.fit_transform(scaled_data)

# === 10. Kết hợp kết quả PCA & t-SNE ===
dim_df = pd.DataFrame({
    "PCA1": pca_result[:, 0],
    "PCA2": pca_result[:, 1],
    "tSNE1": tsne_result[:, 0],
    "tSNE2": tsne_result[:, 1],
    vn30_col: numeric_clean[vn30_col].values
})

# === 11. Biểu đồ tương tác Plotly ===
fig_pca = px.scatter(
    dim_df, x="PCA1", y="PCA2",
    color=vn30_col, color_continuous_scale="Viridis",
    title="🔬 PCA — 2 thành phần chính",
    hover_data=[vn30_col]
)
fig_pca.show()

fig_tsne = px.scatter(
    dim_df, x="tSNE1", y="tSNE2",
    color=vn30_col, color_continuous_scale="Plasma",
    title="🔬 t-SNE — 2 chiều",
    hover_data=[vn30_col]
)
fig_tsne.show()

print("\n Hoàn tất phân tích PCA + t-SNE và lưu file data2.xlsx!")
print(f" Tổng số biểu đồ đã tạo: 6 biểu đồ (2 bar charts + 2 heatmaps + PCA + t-SNE)")
# =====================================================
#  YÊU CẦU 6 — PHÂN CỤM THEO CỔ PHIẾU (gom các biến con)
# =====================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# === 1. Đọc dữ liệu ===
path_data = r"F:\KL\data2.xlsx"
df = pd.read_excel(path_data)
print(f" Đã đọc dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột")

# === 2. Chuẩn hóa tên cột ===
df.columns = [c.strip() for c in df.columns]

# === 3. Trích mã cổ phiếu từ tên cột (ví dụ: GAS_KL, VNM_Thấp, TPB_Mở, ...) ===
def extract_symbol(col):
    m = re.search(r"([A-Z]{2,5})_", col)
    return m.group(1) if m else None

symbols = [extract_symbol(c) for c in df.columns]
df.columns = pd.MultiIndex.from_arrays([symbols, df.columns])

# === 4. Gom theo mã cổ phiếu (trung bình các biến con) ===
grouped = df.groupby(level=0, axis=1).mean()
print(f" Có {grouped.shape[1]} mã cổ phiếu sau khi gom nhóm.")

# === 5. Chuẩn hóa dữ liệu (mỗi hàng = 1 cổ phiếu) ===
scaler = StandardScaler()
scaled_data = scaler.fit_transform(grouped.T)

# === 6. Tìm số cụm tối ưu bằng Silhouette ===
K_range = range(2, 8)
scores = []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    scores.append(score)

best_k = K_range[np.argmax(scores)]
print(f" Số cụm tối ưu (theo Silhouette): {best_k}")

plt.plot(K_range, scores, 'o-')
plt.title("Silhouette Score để chọn số cụm tối ưu")
plt.xlabel("Số cụm (K)")
plt.ylabel("Silhouette Score")
plt.show()

# === 7. KMeans chính thức ===
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(scaled_data)

cluster_df = pd.DataFrame({
    'Cổ phiếu': grouped.columns,
    'Cluster': labels
}).sort_values('Cluster')

# === 8. Trực quan hoá PCA 2D ===
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster'] = labels
pca_df['Cổ phiếu'] = grouped.columns

fig = px.scatter(
    pca_df, x='PC1', y='PC2',
    color=pca_df['Cluster'].astype(str),
    text='Cổ phiếu',
    title=f" Phân cụm cổ phiếu theo hành vi giá (K-Means, K={best_k})",
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_traces(textposition='top center', marker=dict(size=8))
fig.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(legend_title_text='Cụm cổ phiếu', title_x=0.5)
fig.show()

# === 9. Lưu file Excel ===
out_path = r"F:\KL\K-mean.xlsx"
cluster_df.to_excel(out_path, index=False)
print(f" Đã lưu kết quả tại: {out_path}")

# === 10. Diễn giải ===
for c in range(best_k):
    members = cluster_df[cluster_df['Cluster']==c]['Cổ phiếu'].tolist()
    print(f"\n🔹 Cụm {c} ({len(members)} cổ phiếu): {', '.join(members)}")

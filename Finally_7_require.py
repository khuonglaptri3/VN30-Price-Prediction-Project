import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.ardl import ARDL
from tensorflow.keras import layers, models, callbacks

# -----------------------------
# 1 LOAD DATA & BASIC CLEAN
# -----------------------------
path = r"F:\KL\PCA.xlsx"
df = pd.read_excel(path)

df['Ngày'] = pd.to_datetime(df['Ngày'])
df = df.sort_values('Ngày')

target_col = [c for c in df.columns if "VN30" in c and "Lần cuối" in c][0]
feature_cols = [c for c in df.columns if c not in ['Ngày', target_col]]

# -----------------------------
#  TẠO TẬP TRAIN / TEST THEO NGÀY
# -----------------------------
train_start = "2020-01-01"
train_end   = "2025-08-20"
test_start  = "2025-09-01"
test_end    = "2025-09-22"

train_df = df[(df['Ngày'] >= train_start) & (df['Ngày'] <= train_end)].copy()
test_df  = df[(df['Ngày'] >= test_start)  & (df['Ngày'] <= test_end)].copy()

X_train, y_train = train_df[feature_cols], train_df[target_col]
X_test, y_test   = test_df[feature_cols],  test_df[target_col]

print(f"Train: {len(X_train)} mẫu, Test: {len(X_test)} mẫu")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------
#  ARDL MODEL
# -----------------------------
def run_ardl(y, X, max_lags):
    model = ARDL(endog=y, exog=X, lags=max_lags).fit()
    print(model.summary())

    sig_vars = model.pvalues[model.pvalues <= 0.1].index.tolist()
    print(f"\n📉 Biến có ý nghĩa thống kê (p ≤ 0.1): {sig_vars}")

    y_pred = model.predict()
    rmse = np.sqrt(mean_squared_error(y[max_lags:], y_pred[max_lags:]))
    mae  = mean_absolute_error(y[max_lags:], y_pred[max_lags:])
    mape = mean_absolute_percentage_error(y[max_lags:], y_pred[max_lags:])
    print(f"ARDL RMSE={rmse:.4f} | MAE={mae:.4f} | MAPE={mape:.4f}")
    return rmse, mae, mape, model

p = int(input("Nhập độ trễ tối ưu cho ARDL (ví dụ 1 hoặc 2): "))
ardl_rmse, ardl_mae, ardl_mape, ardl_model = run_ardl(y_train, X_train, max_lags=p)

# -----------------------------
#  TRUYỀN THỐNG & MÁY HỌC
# -----------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)

xgb = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.9,
    colsample_bytree=0.9, random_state=42
)
xgb.fit(X_train_scaled, y_train)
xgb_pred = xgb.predict(X_test_scaled)

linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)
linreg_pred = linreg.predict(X_test_scaled)

lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_pred = lasso.predict(X_test_scaled)

# -----------------------------
#  LSTM & GRU (Deep Learning)
# -----------------------------
def build_lstm(n_features):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(1, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(n_features):
    model = models.Sequential([
        layers.GRU(64, return_sequences=True, input_shape=(1, n_features)),
        layers.Dropout(0.2),
        layers.GRU(32),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

X_train_seq = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_seq  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

lstm = build_lstm(X_train_scaled.shape[1])
gru  = build_gru(X_train_scaled.shape[1])

es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

lstm.fit(X_train_seq, y_train, epochs=100, batch_size=16, verbose=0, validation_split=0.1, callbacks=[es])
gru.fit(X_train_seq, y_train, epochs=100, batch_size=16, verbose=0, validation_split=0.1, callbacks=[es])

lstm_pred = lstm.predict(X_test_seq).ravel()
gru_pred  = gru.predict(X_test_seq).ravel()

# -----------------------------
#  ĐÁNH GIÁ MÔ HÌNH
# -----------------------------
def evaluate_model(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE": mape}

results = []
results.append(evaluate_model(y_test, linreg_pred, "Linear Regression"))
results.append(evaluate_model(y_test, lasso_pred, "Lasso Regression"))
results.append(evaluate_model(y_test, rf_pred, "Random Forest"))
results.append(evaluate_model(y_test, xgb_pred, "XGBoost"))
results.append(evaluate_model(y_test, lstm_pred, "LSTM"))
results.append(evaluate_model(y_test, gru_pred, "GRU"))

# ARDL dự báo cho tập test
ardl_forecast = ardl_model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1, exog_oos=X_test)
results.append(evaluate_model(y_test, ardl_forecast, "ARDL"))

result_df = pd.DataFrame(results).sort_values("RMSE")
print("\n KẾT QUẢ SO SÁNH CÁC MÔ HÌNH:")
print(result_df)

best_model = result_df.iloc[0]["Model"]
print(f"\n🏆 Mô hình tốt nhất: {best_model}")

# -----------------------------
# 7️⃣ TRỰC QUAN HÓA KẾT QUẢ
# -----------------------------
pred_map = {
    "Linear Regression": linreg_pred,
    "Lasso Regression": lasso_pred,
    "ARDL": ardl_forecast,
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "LSTM": lstm_pred,
    "GRU": gru_pred
}

plt.figure(figsize=(12,6))
plt.plot(test_df['Ngày'], y_test.values, label='Thực tế', color='black', linewidth=2)
for name, preds in pred_map.items():
    plt.plot(test_df['Ngày'], preds, label=name, alpha=0.8)
plt.title("So sánh dự báo VN30_Lần cuối giữa các mô hình", fontsize=13)
plt.xlabel("Ngày"); plt.ylabel("VN30_Lần cuối")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Biểu đồ thanh so sánh RMSE ---
plt.figure(figsize=(8,5))
sns.barplot(data=result_df, x="RMSE", y="Model", palette="viridis")
plt.title("So sánh sai số (RMSE) giữa các mô hình")
plt.xlabel("RMSE (Càng thấp càng tốt)")
plt.ylabel("Mô hình")
plt.tight_layout()
plt.show()

# --- Vẽ riêng mô hình tốt nhất ---
best_pred = pred_map[best_model]
plt.figure(figsize=(10,5))
plt.plot(test_df['Ngày'], y_test.values, label="Thực tế", color="black", linewidth=2)
plt.plot(test_df['Ngày'], best_pred, label=f"{best_model} (Best)", color="teal", linewidth=2)
plt.title(f"Kết quả mô hình tốt nhất: {best_model}")
plt.xlabel("Ngày"); plt.ylabel("VN30_Lần cuối")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

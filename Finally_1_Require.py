import os
import re
import pandas as pd

# ======= cấu hình =======
input_folder = r"F:\KL\csv_folder"    
# Thư mục chứa các CSV
master_filename = 'VN30.csv'                    
# File mốc ngày
output_file = os.path.expanduser(r"F:\KL\KQ_VN30.xlsx")

# Các cột cần lấy & thứ tự
COLS = ['Lần cuối', 'Mở', 'Cao', 'Thấp', 'KL', '% Thay đổi']

# ---------- Helpers làm sạch ----------
def to_number(x):
    """Chuyển chuỗi số có dấu phẩy ngăn cách nghìn về float."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    s = s.replace(',', '')
    # Nếu là dạng "1,234.56%" thì để hàm phần trăm xử lý riêng
    try:
        return float(s)
    except:
        return pd.NA

def parse_percent(x):
    """'−0.97%' -> -0.97 (đơn vị %)"""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(',', '')
    s = s.replace('%', '')
    # xử lý dấu âm unicode “−”
    s = s.replace('−', '-')
    try:
        return float(s)
    except:
        return pd.NA

def parse_volume(x):
    """
    '55.92M' -> 55_920_000
    '1.2B'   -> 1_200_000_000
    Nếu không có hậu tố thì cố gắng parse số thường.
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().replace(',', '')
    s = s.replace('−', '-')  # đề phòng
    m = re.fullmatch(r'(-?\d+(\.\d+)?)([KMB])?', s, flags=re.IGNORECASE)
    if m:
        val = float(m.group(1))
        suf = (m.group(3) or '').upper()
        if suf == 'K':
            val *= 1_000
        elif suf == 'M':
            val *= 1_000_000
        elif suf == 'B':
            val *= 1_000_000_000
        return val
    # fallback: cố parse số trần
    try:
        return float(s)
    except:
        return pd.NA

def clean_df(df):
    """Chuẩn hoá 6 cột về numeric theo đúng kiểu."""
    if 'Lần cuối' in df: df['Lần cuối'] = df['Lần cuối'].apply(to_number)
    if 'Mở'       in df: df['Mở']       = df['Mở'].apply(to_number)
    if 'Cao'      in df: df['Cao']      = df['Cao'].apply(to_number)
    if 'Thấp'     in df: df['Thấp']     = df['Thấp'].apply(to_number)
    if 'KL'       in df: df['KL']       = df['KL'].apply(parse_volume)
    if '% Thay đổi' in df: df['% Thay đổi'] = df['% Thay đổi'].apply(parse_percent)
    return df

def read_and_standardize(path):
    """Đọc csv, chuẩn hoá cột Ngày và giữ đúng 6 cột nếu có."""
    df = pd.read_csv(path)
    # Chuẩn hoá ngày (d/M/Y)
    df['Ngày'] = pd.to_datetime(df['Ngày'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Ngày'])
    # Chỉ giữ những cột ta cần nếu tồn tại
    keep = ['Ngày'] + [c for c in COLS if c in df.columns]
    df = df[keep]
    df = clean_df(df)
    return df.sort_values('Ngày').reset_index(drop=True)

def add_prefix(df, prefix):
    """Đổi tên 6 cột với tiền tố <prefix>_...; giữ nguyên cột Ngày."""
    rename_map = {c: f'{prefix}_{c}' for c in COLS if c in df.columns}
    return df.rename(columns=rename_map)

# ---------- Pipeline chính ----------
def main():
    # 1) Đọc file mốc VN30 để lấy dãy ngày
    master_path = os.path.join(input_folder, master_filename)
    df_master = read_and_standardize(master_path)

    # Chỉ cần cột Ngày làm khung
    final_df = df_master[['Ngày']].copy()

    # 2) Duyệt tất cả file CSV, gồm cả VN30.csv (để lấy 6 cột của chính VN30)
    for fname in sorted(os.listdir(input_folder)):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(input_folder, fname)
        symbol = os.path.splitext(fname)[0]  # tiền tố

        df = read_and_standardize(path)
        df = add_prefix(df, symbol)

        # Chỉ merge theo mốc Ngày của VN30 (left join)
        final_df = final_df.merge(df, on='Ngày', how='left')

    # 3) Điền thiếu (tuỳ nhu cầu; có thể bỏ nếu muốn giữ NaN)
    for col in final_df.columns:
        if col == 'Ngày':
            continue
        # nội suy tuyến tính cho số; sau đó ffill/bfill
        if pd.api.types.is_numeric_dtype(final_df[col]):
            final_df[col] = final_df[col].interpolate(method='linear')
        final_df[col] = final_df[col].ffill().bfill()

    # 4) Xuất Excel
    final_df.to_excel(output_file, index=False, engine='openpyxl')
    print(f'Đã lưu tại: {output_file}')

if __name__ == '__main__':
    main()

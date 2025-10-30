import pandas as pd
df = pd.read_excel(r'F:\KL\KQ_VN30.xlsx')
print(df.head())
print(df.tail())
print(df.describe())
print(df.info())

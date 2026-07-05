import pandas as pd

df = pd.read_parquet("data/processed/movies.parquet")

print(df.columns)
print(type(df["tokens"].iloc[0]))
print(df["tokens"].iloc[0][:10])
import pandas as pd

df =  pd.read_csv("data/students.csv")
# print(df.head(8))
# print(df.info())
print(df.isna().sum())
score_cols = ["reading", "writing"]
df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
df[score_cols] = df[score_cols].fillna(df[score_cols].mean())
print(df.isna().sum())
print(df.head(8))

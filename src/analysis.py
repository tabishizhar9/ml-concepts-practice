import pandas as pd
import numpy as np

df =  pd.read_csv("data/students.csv")
score_cols = ["reading", "writing"]


# print(df.head(8))
# print(df.info())

#########       practice handling missing values using mean and median
# print(df.isna().sum())
# score_cols = ["reading", "writing"]
# df[score_cols] = df[score_cols].apply(pd.to_numeric, errors="coerce")
# df[score_cols] = df[score_cols].fillna(df[score_cols].mean())
# print(df.isna().sum())
# print(df.head(8))

#########       added a feature/ column average_score
df["avg_score"] = df[score_cols].mean(axis=1)
print(df[["name", "avg_score"]].head())
print(df.head(8))

########        label the data Pass / Fail
df["pass"] = np.where(df["avg_score"] >= 60, "Yes", "No")
print(df[["name", "avg_score", "pass"]].head(8))

########        
print("Average Math score :",df["math"].mean())
print("Average Pass Score :",(df["pass"] == "Yes").mean())

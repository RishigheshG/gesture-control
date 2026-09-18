import pandas as pd

df = pd.read_csv("gesture_data.csv", header=None)
print("total rows:", len(df))
print("columns per row:", df.shape[1], "(should be 64: 63 numbers + 1 label)")
print(df.iloc[:, -1].value_counts())
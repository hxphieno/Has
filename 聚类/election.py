import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("dataset2/pv.csv")
print(df.head())
print(df.columns)

plt.figure(figsize=(12,6))

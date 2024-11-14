import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans

with open('dataset2/power.txt', 'r') as file:
    content = file.read()

content = [i.split(",") for i in content.split("\n")]
df = pd.DataFrame(content).apply(pd.to_numeric, errors='coerce')

df = df.apply(lambda row:
              row.fillna(df.mean(axis=1, skipna=True)), axis=1)

df['mean'] = df.mean(axis=1)
df['var'] = df.drop("mean", axis=1).var(axis=1)

X = df[['mean', 'var']]

if (True):
    dbscan = DBSCAN(eps=5, min_samples=5)
    dbscan.fit(X)
    labels = dbscan.labels_
    df['cluster'] = labels
    outliers = df[df['cluster'] == -1]
else:
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    labels = kmeans.labels_
    df['cluster'] = labels
    df['distance_to_center'] = kmeans.transform(X).min(axis=1)
    outlier_ratio = 0.1
    outliers_per_cluster = int(outlier_ratio * len(df) / kmeans.n_clusters)
    outliers = df.nlargest(outliers_per_cluster, 'distance_to_center')



# 打印异常值
print("异常值：")
print(outliers)

# 根据index获取原文中的异常值

data = pd.read_csv("dataset2/pv.csv")

print("异常原值：")
print(data.loc[outliers.index])

plt.figure(figsize=(10, 6))
for column in data.loc[outliers.index].columns:
    if column != 'time':
        plt.plot(data.loc[outliers.index]['time'], data.loc[outliers.index][column], label=column)

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Power Data Plot for Outliers')
plt.legend()
plt.show()

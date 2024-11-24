import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, KMeans


def analyse_window_data(path, step=1):
    with open(path, 'r') as file:
        content = file.read()

    content = [i.split(",") for i in content.split("\n")]
    df = pd.DataFrame(content).apply(pd.to_numeric, errors='coerce')

    df = df.apply(lambda row:
                  row.fillna(df.mean(axis=1, skipna=True)), axis=1)
    X = df
    df['lag_1_autocorr'] = df.apply(lambda row: row.autocorr(lag=1), axis=1)
    df['mean'] = X.mean(axis=1)
    df['var'] = X.var(axis=1)
    df['skew'] = X.skew(axis=1)
    df['kurt'] = X.kurt(axis=1)
    df['range'] = X.max(axis=1) - X.min(axis=1)

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
        outlier_ratio = 0.05
        outliers_per_cluster = int(outlier_ratio * len(df) / kmeans.n_clusters)
        outliers = df.nlargest(outliers_per_cluster, 'distance_to_center')

    # 打印异常值
    print("异常值：")
    print(outliers)

    # 根据index获取原文中的异常值

    data = pd.read_csv("dataset2/pv.csv")

    print("异常原值：")
    print(data.loc[outliers.index * step])

    plt.figure(figsize=(10, 6))
    for column in data.loc[outliers.index].columns:
        if column != 'time':
            plt.plot(data.loc[outliers.index]['time'], data.loc[outliers.index][column], label=column)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Power Data Plot for Outliers')
    plt.legend()
    plt.show()


analyse_window_data("dataset2/power5.txt")
analyse_window_data("dataset2/power6.txt")
analyse_window_data("dataset2/power7.txt")
analyse_window_data("dataset2/power8.txt")
analyse_window_data("dataset2/power-step2.txt", step=2)
analyse_window_data("dataset2/power-step3.txt", step=2)

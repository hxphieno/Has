import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN,KMeans

# 加载数据
df = pd.read_csv('dataset2/pv.csv', parse_dates=['time'], index_col='time')

df = df.dropna()







# # DBSCAN
# db = DBSCAN(eps=4, min_samples=2).fit(df)
# df['cluster'] = db.labels_
#
# # 筛选出异常点（标签为-1的样本）
# outliers = df[df['cluster'] == -1]
#
# print("异常点:")
# print(outliers)
#
# # 可视化数据和异常点
# plt.figure(figsize=(10, 6))  # 设置图表大小
# sns.scatterplot(data=outliers, x='power', y='generation', color='red')  # 使用Seaborn绘制散点图
# plt.title('Outliers in Power and Generation')  # 设置图表标题
# plt.xlabel('Power')  # 设置x轴标签
# plt.ylabel('Generation')  # 设置y轴标签
# plt.show()

#K-means
# kmeans=KMeans(n_clusters=3)
# # 使用 K-means 进行聚类
# df['cluster'] = kmeans.fit_predict(df)
#
# centroids = kmeans.cluster_centers_
#
# # 计算每个样本到质心的距离
# distances = np.linalg.norm(df[['power','generation']].values - centroids[df['cluster']], axis=1)
#
# # 设置一个阈值，认为离群点是距离质心较远的点
# threshold = np.mean(distances) + 20 * np.std(distances)
#
# # 筛选出异常点（距离质心较远的点）
# outliers = df[distances > threshold]
#
# # 输出异常点
# print("异常点:")
# print(outliers)
# print(outliers.shape[0])
#
# # 可视化数据和异常点
# plt.figure(figsize=(10, 6))  # 设置图表大小
# sns.scatterplot(data=outliers, x='power', y='generation', color='red')  # 使用Seaborn绘制散点图
# plt.title('Outliers in Power and Generation')  # 设置图表标题
# plt.xlabel('Power')  # 设置x轴标签
# plt.ylabel('Generation')  # 设置y轴标签
# plt.show()

# plt.figure(figsize=(7, 5))

# # 趋势折线图
# plt.plot(df.index, df['power'], label='Power')
# plt.legend()
# plt.title('Power Trend Line Plot')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()
#
# plt.plot(df.index, df['generation'], label='Generation')
# plt.legend()
# plt.title('Generation Trend Line Plot')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.show()
#
# # 散点图
# plt.scatter(df.index, df['power'], alpha=0.5)
# plt.title('Scatter Plot of Power')
# plt.xlabel('Time')
# plt.ylabel('Power')
# plt.show()
#
# plt.scatter(df.index, df['generation'], alpha=0.5)
# plt.title('Scatter Plot of Generation')
# plt.xlabel('Time')
# plt.ylabel('Generation')
# plt.show()
#
# # 盒图
# plt.figure(figsize=(10, 5))
# sns.boxplot(data=df[['power', 'generation']])
# plt.title('Boxplot of Power and Generation')
# plt.show()

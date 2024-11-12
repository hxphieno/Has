from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集转换为 Pandas DataFrame 方便查看
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df.head()

# KMeans 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 聚类结果
clusters = kmeans.labels_

# 将聚类结果添加到 DataFrame
df['cluster'] = clusters

from sklearn.decomposition import PCA

# 使用 PCA 将特征降维至 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 转换为 DataFrame 方便可视化
df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['cluster'] = clusters
df_pca['species'] = y

# 使用 Seaborn 绘制聚类结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_pca, palette='viridis', s=100)
plt.title('KMeans Clustering of Iris Dataset (PCA Projection)')
plt.show()

from sklearn.metrics import adjusted_rand_score, silhouette_score

# 调整兰德指数（ARI）
ari = adjusted_rand_score(y, clusters)
print(f'Kmeans：Adjusted Rand Index (ARI): {ari:.4f}')

# 轮廓系数
silhouette_avg = silhouette_score(X, clusters)
print(f'Kmeans：Silhouette Score: {silhouette_avg:.4f}')
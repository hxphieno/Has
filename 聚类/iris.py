from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, recall_score, \
    precision_score, davies_bouldin_score

# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据集转换为 Pandas DataFrame 方便查看
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df.head()

# KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
clusters = kmeans.labels_

df['cluster'] = clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_pca = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
df_pca['cluster'] = clusters
df_pca['species'] = y

# 使用 Seaborn 绘制聚类结果
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=df_pca, palette='viridis', s=100)
plt.title('KMeans Clustering of Iris Dataset (PCA Projection)')
plt.show()



silhouette_avg_kmeans = silhouette_score(X, clusters)
db_index_kmeans = davies_bouldin_score(X, clusters)
print(f'KMeans：Silhouette Score: {silhouette_avg_kmeans:.4f}')
print(f'KMeans：Davies-Bouldin Index: {db_index_kmeans:.4f}')

# 基于决策树的层次聚类
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"DecisionTree Accuracy: {accuracy:.2f}")
# 精确率
precision = precision_score(y_test, y_pred, average='weighted')
print(f'DecisionTree Precision: {precision:.4f}')
# 召回率
recall = recall_score(y_test, y_pred, average='weighted')
print(f'DecisionTree Recall: {recall:.4f}')
# F1分数
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'DecisionTree F1 Score: {f1:.4f}')

# 可视化决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree for Iris Dataset")
plt.show()

dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X)

silhouette_avg_dbscan = silhouette_score(X, y_pred)
db_index_dbscan = davies_bouldin_score(X, y_pred)
print(f'DBSCAN：Silhouette Score: {silhouette_avg_dbscan:.4f}')
print(f'DBSCAN：Davies-Bouldin Index: {db_index_dbscan:.4f}')

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('DBSCAN Clustering on Iris Dataset')
plt.show()

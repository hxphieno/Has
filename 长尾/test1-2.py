import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer  # 导入 SimpleImputer

# 读取数据集 (假设数据集为csv格式)
data = pd.read_csv('../dataset/News.csv')

# 查看数据集基本信息
print(data.head())
print(data.info())

# 列出所有非数值型特征
non_numeric_cols = data.select_dtypes(include=['object']).columns

# 使用 LabelEncoder 对所有非数值型特征进行编码
label_encoders = {}
for col in non_numeric_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# 使用 SimpleImputer 填充缺失值
imputer = SimpleImputer(strategy='mean')  # 对于数值型数据，使用均值填充
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 查看数据填充后的前几行
print(data_imputed.head())

# 假设 'category' 是葡萄酒产区，'type' 是葡萄酒类型，其他列为特征
X = data_imputed.drop(columns=['category'])
y_category = data_imputed['category']

# 数据分割，分别按葡萄酒产区和类型进行分类
X_train, X_test, y_category_train, y_category_test = train_test_split(X, y_category, test_size=0.3, random_state=42)


# 定义分类算法
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    # 'LogisticRegression': LogisticRegression( random_state=42),
    # 'SVM': SVC(random_state=42),  # 支持向量机
    # 'KNN': KNeighborsClassifier(),  # K近邻
    # 'DecisionTree': DecisionTreeClassifier(random_state=42)  # 决策树
}


# 训练并评估模型
def evaluate_model(X_train, X_test, y_train, y_test, models):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
    return results


# 原始准确率评估
category_results = evaluate_model(X_train, X_test, y_category_train, y_category_test, models)


# 输出原始模型评估结果
print("category Classification Accuracy:")
print(category_results)



# 类别重采样：基于葡萄酒产区进行过采样
def resample_data(X_train, y_train):
    # 将数据与标签合并
    data = pd.concat([X_train, y_train], axis=1)

    # 分开少数类和多数类
    majority_class = data[data[y_train.name] == data[y_train.name].mode()[0]]
    minority_class = data[data[y_train.name] != data[y_train.name].mode()[0]]

    # 过采样少数类
    minority_upsampled = resample(minority_class,
                                  replace=True,
                                  n_samples=majority_class.shape[0],
                                  random_state=42)

    # 合并重新构建数据集
    upsampled_data = pd.concat([majority_class, minority_upsampled])

    # 返回特征和标签
    return upsampled_data.drop(columns=[y_train.name]), upsampled_data[y_train.name]

def smote_resample(X_train, y_train):
    # 初始化SMOTE对象
    smote = SMOTE(random_state=42)

    # 对训练数据进行过采样
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # 返回过采样后的数据
    return X_resampled, y_resampled

# 进行重采样
X_train_resampled, y_train_resampled = smote_resample(X_train, y_category_train)


# 重新评估重采样后的模型
category_resampled_results = evaluate_model(X_train_resampled, X_test, y_train_resampled, y_category_test, models)

# 输出重采样后结果
print("category Classification Accuracy (After Resampling):")
print(category_resampled_results)




# 可视化分类准确率
def plot_accuracy(results, title):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.show()


# 可视化结果
plot_accuracy(category_results, 'Original category Classification Accuracy')
plot_accuracy(category_resampled_results, 'category Classification Accuracy after Resampling')


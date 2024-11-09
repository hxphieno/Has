import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
data = pd.read_csv('dataset/diabetes.csv')

# 查看数据集基本信息
print(data.head())
print(data.info())

# 使用 SimpleImputer 填充缺失值
imputer = SimpleImputer(strategy='most_frequent')  # 对于数值型数据，使用均值填充
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 查看数据填充后的前几行
print(data_imputed.head())

X = data_imputed.drop(columns=['Pregnancies', 'Age'])
y_pregnancies = data_imputed['Pregnancies']
y_age = data_imputed['Age']

# 数据分割，分别按葡萄酒产区和类型进行分类
X_train, X_test, y_pregnancies_train, y_pregnancies_test = train_test_split(X, y_pregnancies, test_size=0.3, random_state=42)
X_train_age, X_test_age, y_age_train, y_age_test = train_test_split(X, y_age, test_size=0.3, random_state=42)

# 定义分类算法
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42),
    'SVM': SVC(random_state=42),  # 支持向量机
    'DecisionTree': DecisionTreeClassifier(random_state=42)  # 决策树
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
pregnancies_results = evaluate_model(X_train, X_test, y_pregnancies_train, y_pregnancies_test, models)
age_results = evaluate_model(X_train_age, X_test_age, y_age_train, y_age_test, models)

# 输出原始模型评估结果
print("pregnancies Classification Accuracy:")
print(pregnancies_results)
print("age Classification Accuracy:")
print(age_results)


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
                                  n_samples=6 * majority_class.shape[0],
                                  random_state=42)

    # 合并重新构建数据集
    upsampled_data = pd.concat([majority_class, minority_upsampled])

    # 返回特征和标签
    return upsampled_data.drop(columns=[y_train.name]), upsampled_data[y_train.name]


# 进行重采样
X_train_resampled, y_train_resampled = resample_data(X_train, y_pregnancies_train)
X_train_age_resampled, y_age_train_resampled = resample_data(X_train_age, y_age_train)

# 重新评估重采样后的模型
pregnancies_resampled_results = evaluate_model(X_train_resampled, X_test, y_train_resampled, y_pregnancies_test, models)
age_resampled_results = evaluate_model(X_train_age_resampled, X_test_age, y_age_train_resampled, y_age_test,
                                        models)

# 输出重采样后结果
print("pregnancies Classification Accuracy (After Resampling):")
print(pregnancies_resampled_results)
print("age Classification Accuracy (After Resampling):")
print(age_resampled_results)


from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_pregnancies_train), y=y_pregnancies_train)
class_weights_dict = dict(enumerate(class_weights))

models_with_weights = {
    'RandomForest': RandomForestClassifier(random_state=2),
    'LogisticRegression': LogisticRegression(random_state=32),
    'SVM': SVC(random_state=22),  # 支持向量机
    'DecisionTree': DecisionTreeClassifier(random_state=12)  # 决策树
}


# 重加权后的评估
pregnancies_weighted_results = evaluate_model(X_train, X_test, y_pregnancies_train, y_pregnancies_test, models_with_weights)
age_weighted_results = evaluate_model(X_train_age, X_test_age, y_age_train, y_age_test, models_with_weights)

# 输出重加权后结果
print("pregnancies Classification Accuracy (After Class Weighting):")
print(pregnancies_weighted_results)
print("age Classification Accuracy (After Class Weighting):")
print(age_weighted_results)


# 合并结果为一个 DataFrame，用于方便可视化
def create_comparison_dataframe(pregnancies_results, pregnancies_resampled_results, pregnancies_weighted_results, age_results,
                                age_resampled_results, age_weighted_results):
    pregnancies_comparison = pd.DataFrame({
        'Model': pregnancies_results.keys(),
        'Original': pregnancies_results.values(),
        'Resampled': pregnancies_resampled_results.values(),
        'Weighted': pregnancies_weighted_results.values()
    })

    age_comparison = pd.DataFrame({
        'Model': age_results.keys(),
        'Original': age_results.values(),
        'Resampled': age_resampled_results.values(),
        'Weighted': age_weighted_results.values()
    })

    return pregnancies_comparison, age_comparison


# 创建对比的 DataFrame
pregnancies_comparison, age_comparison = create_comparison_dataframe(
    pregnancies_results, pregnancies_resampled_results, pregnancies_weighted_results,
    age_results, age_resampled_results, age_weighted_results
)


# 可视化区域分类结果
def plot_comparison(df, title):
    df.set_index('Model', inplace=True)
    df.plot(kind='bar', figsize=(10, 6))
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# 绘制区域分类结果对比图
plot_comparison(pregnancies_comparison, 'pregnancies Classification Accuracy Comparison')

# 绘制葡萄酒类型分类结果对比图
plot_comparison(age_comparison, ' age Classification Accuracy Comparison')

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
data = pd.read_csv('../dataset/StudentPerformanceFactors.csv')

non_numeric_cols = data.select_dtypes(include=['object']).columns

# 使用 LabelEncoder 对所有非数值型特征进行编码
label_encoders = {}
for col in non_numeric_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# 使用 SimpleImputer 填充缺失值
imputer = SimpleImputer(strategy='most_frequent')  # 对于数值型数据，使用均值填充
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 查看数据填充后的前几行
print(data_imputed.head())

X = data_imputed.drop(columns=['Hours_Studied', 'Tutoring_Sessions'])
y_Hours_Studied = data_imputed['Hours_Studied']
y_Tutoring_Sessions = data_imputed['Tutoring_Sessions']

# 数据分割，分别按葡萄酒产区和类型进行分类
X_train, X_test, y_Hours_Studied_train, y_Hours_Studied_test = train_test_split(X, y_Hours_Studied, test_size=0.3, random_state=42)
X_train_Tutoring_Sessions, X_test_Tutoring_Sessions, y_Tutoring_Sessions_train, y_Tutoring_Sessions_test = train_test_split(X, y_Tutoring_Sessions, test_size=0.3, random_state=42)

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
Hours_Studied_results = evaluate_model(X_train, X_test, y_Hours_Studied_train, y_Hours_Studied_test, models)
Tutoring_Sessions_results = evaluate_model(X_train_Tutoring_Sessions, X_test_Tutoring_Sessions, y_Tutoring_Sessions_train, y_Tutoring_Sessions_test, models)

# 输出原始模型评估结果
print("Hours_Studied Classification Accuracy:")
print(Hours_Studied_results)
print("Tutoring_Sessions Classification Accuracy:")
print(Tutoring_Sessions_results)


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
X_train_resampled, y_train_resampled = resample_data(X_train, y_Hours_Studied_train)
X_train_Tutoring_Sessions_resampled, y_Tutoring_Sessions_train_resampled = resample_data(X_train_Tutoring_Sessions, y_Tutoring_Sessions_train)

# 重新评估重采样后的模型
Hours_Studied_resampled_results = evaluate_model(X_train_resampled, X_test, y_train_resampled, y_Hours_Studied_test, models)
Tutoring_Sessions_resampled_results = evaluate_model(X_train_Tutoring_Sessions_resampled, X_test_Tutoring_Sessions, y_Tutoring_Sessions_train_resampled, y_Tutoring_Sessions_test,
                                        models)

# 输出重采样后结果
print("Hours_Studied Classification Accuracy (After Resampling):")
print(Hours_Studied_resampled_results)
print("Tutoring_Sessions Classification Accuracy (After Resampling):")
print(Tutoring_Sessions_resampled_results)


from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_Hours_Studied_train), y=y_Hours_Studied_train)
class_weights_dict = dict(enumerate(class_weights))

models_with_weights = {
    'RandomForest': RandomForestClassifier(random_state=2),
    'LogisticRegression': LogisticRegression(random_state=32),
    'SVM': SVC(random_state=22),  # 支持向量机
    'DecisionTree': DecisionTreeClassifier(random_state=12)  # 决策树
}


# 重加权后的评估
Hours_Studied_weighted_results = evaluate_model(X_train, X_test, y_Hours_Studied_train, y_Hours_Studied_test, models_with_weights)
Tutoring_Sessions_weighted_results = evaluate_model(X_train_Tutoring_Sessions, X_test_Tutoring_Sessions, y_Tutoring_Sessions_train, y_Tutoring_Sessions_test, models_with_weights)

# 输出重加权后结果
print("Hours_Studied Classification Accuracy (After Class Weighting):")
print(Hours_Studied_weighted_results)
print("Tutoring_Sessions Classification Accuracy (After Class Weighting):")
print(Tutoring_Sessions_weighted_results)


# 合并结果为一个 DataFrame，用于方便可视化
def create_comparison_dataframe(Hours_Studied_results, Hours_Studied_resampled_results, Hours_Studied_weighted_results, Tutoring_Sessions_results,
                                Tutoring_Sessions_resampled_results, Tutoring_Sessions_weighted_results):
    Hours_Studied_comparison = pd.DataFrame({
        'Model': Hours_Studied_results.keys(),
        'Original': Hours_Studied_results.values(),
        'Resampled': Hours_Studied_resampled_results.values(),
        'Weighted': Hours_Studied_weighted_results.values()
    })

    Tutoring_Sessions_comparison = pd.DataFrame({
        'Model': Tutoring_Sessions_results.keys(),
        'Original': Tutoring_Sessions_results.values(),
        'Resampled': Tutoring_Sessions_resampled_results.values(),
        'Weighted': Tutoring_Sessions_weighted_results.values()
    })

    return Hours_Studied_comparison, Tutoring_Sessions_comparison


# 创建对比的 DataFrame
Hours_Studied_comparison, Tutoring_Sessions_comparison = create_comparison_dataframe(
    Hours_Studied_results, Hours_Studied_resampled_results, Hours_Studied_weighted_results,
    Tutoring_Sessions_results, Tutoring_Sessions_resampled_results, Tutoring_Sessions_weighted_results
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
plot_comparison(Hours_Studied_comparison, 'Hours_Studied Classification Accuracy Comparison')

# 绘制葡萄酒类型分类结果对比图
plot_comparison(Tutoring_Sessions_comparison, ' Tutoring_Sessions Classification Accuracy Comparison')

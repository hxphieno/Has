import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv("StudentPerformanceFactors.csv")
print(df.head())

label_encoders = {}
for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

X = df.drop(["Exam_Score","Gender"], axis=1)
y = df["Exam_Score"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [
    ('gb', GradientBoostingRegressor(learning_rate=0.03,max_depth=2,n_estimators=1500,min_samples_leaf=3,min_samples_split=8)),
    ('xgb', XGBRegressor(learning_rate=0.03,max_depth=2,n_estimators=1500))
]

# 定义元模型
final_estimator = LinearRegression()

# 创建堆叠模型
model_stack = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)

# model = XGBRegressor(learning_rate=0.03,max_depth=2,n_estimators=1500)
cv_scores = cross_val_score(model_stack, X, y, cv=5, scoring='r2')

# 打印每次交叉验证的R²分数
print(f'CV R² scores: {cv_scores}')
print(f'CV R² average: {cv_scores.mean()}')
print(f'CV R² std: {cv_scores.std()}')

# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)
# print(f'R² (Coefficient of determination): {r2}')
#
# from sklearn.metrics import mean_squared_error
#
# mse = mean_squared_error(y_test, y_pred)
# print(f'MSE (Coefficient of determination): {mse}')
#
# from math import sqrt
#
# rmse = sqrt(mse)
# print(f'RMSE (Coefficient of determination): {rmse}')
#
# from sklearn.metrics import mean_absolute_error
#
# mae = mean_absolute_error(y_test, y_pred)
# print(f'MAE (Coefficient of determination): {mae}')

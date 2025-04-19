import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. 讀取資料
df = pd.read_csv('Taipei_house.csv')

# 2. 資料預處理
# 填補缺失值
# Only fill numeric columns with the mean
numerical_columns = ['土地面積', '建物總面積', '屋齡', '樓層', '總樓層', '房數', '廳數', '衛數', '電梯', '經度', '緯度', '總價'] # Include '總價' here
for column in numerical_columns:
    df[column].fillna(df[column].mean(), inplace=True)  

# 3. 特徵工程
X = df.drop(columns=['總價'])  # 移除目標欄位
y = df['總價']  # 目標是總價

# 使用Pipeline進行預處理和建模
categorical_columns = ['行政區', '用途', '車位類別']  # 類別變數

# 定義預處理步驟
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_columns[:-1]),  # 數值型變數填補, exclude '總價'
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # 類別型變數 One-Hot 編碼
    ])

# 創建Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())  # 這裡我們使用線性回歸，你也可以換成隨機森林、XGBoost等
])

# 4. 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 訓練模型
model.fit(X_train, y_train)

# 6. 預測與評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f"均方誤差 (MSE): {mse:.2f}")
print(f"第一筆預測房價：{y_pred[0]:,.0f} 元")

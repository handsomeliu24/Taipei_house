import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 讀取資料
df = pd.read_csv('taipei_housing.csv')

# 2. 資料處理（示意）
df = df.dropna()
df = pd.get_dummies(df, columns=['區域'], drop_first=True)

# 3. 特徵與目標值
X = df.drop(columns=['建物總價'])
y = df['建物總價']

# 4. 切分資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 建立模型
model = LinearRegression()
model.fit(X_train, y_train)

# 6. 預測與評估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方誤差 MSE: {mse:.2f}")

# 7. 輸入樣本進行預測
sample = X.iloc[0:1]
predicted_price = model.predict(sample)
print(f"預測房價：{predicted_price[0]:,.0f} 元")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler

# 1. Đọc file CSV chứa dữ liệu thật
df = pd.read_csv("vietnam_housing_dataset_filtered_hanoi.csv")

# 2. Lọc dữ liệu chỉ cho "Đống Đa"
dc = df[df["Address"].str.contains("Đống Đa", na=False)]

# 3. Xoá những cột không cần thiết và xử lý giá trị không phải số
dc = dc.drop(columns=['Address', 'House direction', 'Balcony direction', 'Legal status', 'Furniture state'])

# 4. Xử lý giá trị thiếu cho các cột số
numeric_cols = dc.select_dtypes(include=[np.number]).columns
dc[numeric_cols] = dc[numeric_cols].fillna(dc[numeric_cols].mean())

# 5. Tách biến độc lập (X) và biến phụ thuộc (y)
X = dc[['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']]
y = dc['Price']

# 6. Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# 7. Chuẩn hóa các biến độc lập
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# 8. Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_filtered, test_size=0.2, random_state=42)

# 9. Khởi tạo và huấn luyện mô hình MLP (Neural Network)
# Ẩn 100 nơ-ron trong mỗi lớp ẩn, tối đa 2000 lần lặp, alpha là hệ số regularization
mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100),activation='tanh', max_iter=500, alpha=0.9, random_state=42) 
#‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
mlp_model.fit(X_train, y_train)

# 10. Dự đoán trên tập kiểm tra
y_pred = mlp_model.predict(X_test)

# 11. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 13. Chuẩn hóa dữ liệu ví dụ và dự đoán



def train_and_predict_neural(X_train, y_train, X_test):
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100),activation='tanh', max_iter=500, alpha=0.1, random_state=42) 
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    return mlp_model, y_pred

# Hàm đánh giá mô hình Neural Network
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

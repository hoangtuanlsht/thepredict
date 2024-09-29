from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

# Import các mô hình từ các tệp riêng biệt
from linear import train_and_predict_linear, evaluate_model as eval_linear
from neural import train_and_predict_neural, evaluate_model as eval_neural
from lassor import train_and_predict_lassor, evaluate_model as eval_lassor

app = Flask(__name__)

# Đọc và chuẩn bị dữ liệu
df = pd.read_csv("vietnam_housing_dataset_filtered_hanoi.csv")
dc = df[df["Address"].str.contains("Đống Đa", na=False)]
dc = dc.drop(columns=['Address', 'House direction', 'Balcony direction', 'Legal status', 'Furniture state'])
numeric_cols = dc.select_dtypes(include=[np.number]).columns
dc[numeric_cols] = dc[numeric_cols].fillna(dc[numeric_cols].mean())

# Biến độc lập và biến phụ thuộc
X = dc[['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms']]
y = dc['Price']

# Tính Z-score và loại bỏ các outliers
z_scores = np.abs(stats.zscore(X))
filtered_entries = (z_scores < 3).all(axis=1)
X_filtered = X[filtered_entries]
y_filtered = y[filtered_entries]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_filtered)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_filtered, test_size=0.2, random_state=42)
X_train = X_train
y_train = y_train
X_test_scaled = X_test

# Khởi tạo DataFrame toàn cục để lưu dữ liệu nhập vào từ form
Nha_vi_du = pd.DataFrame(columns=['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms'])

@app.route("/", methods=["POST", "GET"])
def index():
    global Nha_vi_du  # Thông báo rằng bạn muốn sử dụng biến toàn cục 'Nha_vi_du'
    
    gia_du_doan = None
    mse = None
    r2 = None
    mae = None

    if request.method == "POST":
        # Lấy giá trị đầu vào từ form
        Area = request.form["area"]
        Frontage = request.form["frontage"]
        AccessRoad = request.form["accessroad"]
        Floors = request.form["floors"]
        Bedrooms = request.form["bedrooms"]
        Bathrooms = request.form["bathrooms"]
        selected_model = request.form["model_type"]

        # Tạo DataFrame mới với giá trị từ form
        new_row = pd.DataFrame({
            'Area': [int(Area)],  # Đặt giá trị vào danh sách
            'Frontage': [int(Frontage)],
            'Access Road': [int(AccessRoad)],
            'Floors': [int(Floors)],
            'Bedrooms': [int(Bedrooms)],
            'Bathrooms': [int(Bathrooms)]
        })

        # Chuẩn hóa dữ liệu mới nhập vào
        new_row_scaled = scaler.transform(new_row)

        # Cập nhật DataFrame toàn cục Nha_vi_du
        Nha_vi_du = pd.concat([Nha_vi_du, new_row], ignore_index=True)

        # Kiểm tra mô hình được chọn và thực hiện dự đoán
        if selected_model == "linear":
            model, y_pred = train_and_predict_linear(X_train, y_train, X_test_scaled)
            gia_du_doan = y_pred[0]
            mse, r2, mae = eval_linear(y_test, model.predict(X_test_scaled))

        elif selected_model == "neural":
            model, y_pred = train_and_predict_neural(X_train, y_train, X_test_scaled)
            gia_du_doan = y_pred[0]
            mse, r2, mae = eval_neural(y_test, model.predict(X_test_scaled))

        elif selected_model == "lassor":
            model, y_pred = train_and_predict_lassor(X_train, y_train, X_test_scaled)
            gia_du_doan = y_pred[0]
            mse, r2, mae = eval_lassor(y_test, model.predict(X_test_scaled))
        return render_template("indes.html", gia_du_doan=gia_du_doan, mse=mse, r2=r2, mae=mae)

    return render_template("indes.html", gia_du_doan=gia_du_doan, mse=mse, r2=r2, mae=mae)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))


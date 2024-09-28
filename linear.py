
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler
from flask import Flask,redirect,url_for, render_template, request,session,flash
from datetime import timedelta


# Load model tại đây khi ứng dụng khởi chạy

app=Flask(__name__)
app.config["SECRET_KEY"] = "cc"
app.permanent_session_lifetime = timedelta(minutes=1)

Area = None
Frontage = None
AccessRoad = None
Floors = None
Bedrooms = None
Bathrooms = None

# @app.route('/login', methods=["POST", "GET"])
# def home():
#     if request.method == "POST":
#         user_name = request.form["name"]
#         session.permanent= True
#         if user_name:
#             session["user"] = user_name
#             flash("You logged in successfully!","info")
#             return render_template("user.html", user=user_name)
#     if "user" in session:
#         name = session["user"]
#         flash("You have already logged in!","info")
#         return render_template("user.html",user=name)
#     return render_template('login.html')
    

# @app.route('/admin')
# def hello_admin():
#     return "<h1> Hello admin dep trai!</h1>"

# @app.route('/user')
# def hello_user():
#     if "user" in session:
#         name = session["user"]
#         return render_template("user.html",user=name)
#     else:
#         return redirect(url_for("home"))
# @app.route("/logout")
# def log_out():
#     flash("You logged out successfully!","info")
#     session.pop("user",None)
#     return redirect(url_for("home"))


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

# 9. Khởi tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# 10. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# 11. Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


# 12. Dự đoán giá nhà cho ví dụ ngẫu nhiên
Nha_vi_du = pd.DataFrame({
    'Area': [],
    'Frontage': [],
    'Access Road': [],
    'Floors': [],
    'Bedrooms': [],
    'Bathrooms': []
})
@app.route("/", methods=["POST", "GET"])
def test():
    global Area, Frontage, AccessRoad, Floors, Bedrooms, Bathrooms,gia_du_doan,model,mse,mae,r2
    gia_du_doan=None
    if request.method == "POST":
        Area=request.form["area"]
        Frontage=request.form["frontage"]
        AccessRoad=request.form["accessroad"]
        Floors=request.form["floors"]
        Bedrooms=request.form["bedrooms"]
        Bathrooms=request.form["bathrooms"]
        selected_model = request.form["model_type"]
        new_row = {
            'Area': int(Area),
            'Frontage': int(Frontage),
            'Access Road': int(AccessRoad),
            'Floors': int(Floors),
            'Bedrooms': int(Bedrooms),
            'Bathrooms': int(Bathrooms)
            }
        Nha_vi_du.loc[len(Nha_vi_du)] = new_row
        print(Nha_vi_du)
        # Xử lý logic dựa trên model được chọn
        if selected_model == "linear":
            model_name = "Linear Regression"
            Nha_vi_du_scaled = scaler.transform(Nha_vi_du)
            gia_du_doan = model.predict(Nha_vi_du_scaled)[-1]
            model_coef = model.coef_
        
            return render_template("indes.html", gia_du_doan=gia_du_doan, model_coef=model_coef, intercept=model.intercept_,
                mse=mse,
                mae=mae,
                r2=r2
            ) 
        # elif selected_model == "lassor":
        #     model_name = "Lasso Regression"
        # elif selected_model == "neural":
        #     model_name = "Neural Network"
            
        # elif selected_model == "ensemble":
        #     model_name = "Stacking Model"
        # Thêm giá trị vào DataFrame
        
    return render_template("indes.html")
    

if __name__=="__main__":
    app.run(debug=True)

# 13. Chuẩn hóa dữ liệu ví dụ và dự đoán




def train_and_predict_linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# Hàm đánh giá mô hình Linear Regression
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler


def train_and_predict_lassor(X_train, y_train, X_test):
    model = Lasso(alpha=0.0001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# Hàm đánh giá mô hình Linear Regression
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

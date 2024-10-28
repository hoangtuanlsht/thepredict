import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler


def train_and_predict_neural(X_train, y_train, X_test):
    mlp_model = MLPRegressor(hidden_layer_sizes=(100, 100),activation='tanh', max_iter=500, alpha=0.9, random_state=42) 
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    return mlp_model, y_pred

# Hàm đánh giá mô hình Neural Network
def evaluate_model(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae

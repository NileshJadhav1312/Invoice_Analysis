"""
Model training and evaluation routines for Freight Cost Prediction.
Provides training wrappers for Linear Regression, Decision Trees, and Random Forests.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_LinearRegression(X_train, y_train):
    """
    Train an Ordinary Least Squares Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_DecisionTree(X_train, y_train, max_depth=5):
    """
    Train a Decision Tree Regressor with specified maximum depth.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_RandomForest(X_train, y_train, max_depth=6):
    """
    Train an ensemble Random Forest Regressor.
    """
    model = RandomForestRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate regression model using Mean Absolute Error (MAE) and RMSE.

    Returns:
    --------
    dict: Performance metrics including MAE and RMSE.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
    }
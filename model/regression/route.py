from flask import  render_template, redirect, url_for, request, current_app,session
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import matplotlib.pyplot as plt
import io
import os
import base64
import pandas as pd
import numpy as np

from . import regression

@regression.route('/regression_type', methods=['GET', 'POST'])
def regression_type():
    if request.method == 'POST':
        regression_type = request.form['model']
        session['regression_model'] = regression_type
        return redirect(url_for('regression.regression_upload'))
    return render_template('regression_type.html')

@regression.route('/regression_upload', methods=['GET', 'POST'])
def regression_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('regression.regression_display', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('regression_upload.html')

@regression.route('/regression_display/<filename>', methods=['GET', 'POST'])
def regression_display(filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)

        if data.isnull().values.any():
            return "Dataset contains missing values. Please handle missing values first."
        
        if not data.select_dtypes(include='object').columns.empty:
            return "Dataset contains categorical columns. Please encode categorical columns first."

        data_html = data.head(10).to_html(index=False)

        return render_template('regression_display.html', tables=[data_html], titles=data.columns.values)
    except Exception as e:
        return str(e)

@regression.route('/regression_eval/<filename>', methods=['GET'])
def regression_eval(filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1]  
        y = data.iloc[:, -1]   
        model = session['regression_model']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "XGBoost Regressor": XGBRegressor(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Support Vector Regressor": SVR()
        }
        
        selected_model = models[model]
        selected_model.fit(X_train, y_train)

        
        y_pred = selected_model.predict(X_test)

        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse) 

        # 📊 Plotting predicted vs actual values
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
        plt.title(f"{model} - Predicted vs Actual")
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        
        # Convert plot to base64 for HTML rendering
        pred_vs_actual = io.BytesIO()
        plt.savefig(pred_vs_actual, format='png')
        pred_vs_actual.seek(0)
        pred_vs_actual_url = base64.b64encode(pred_vs_actual.getvalue()).decode('utf-8')
        plt.close()

        # 📊 Plotting residuals
        plt.figure(figsize=(8, 5))
        residuals = y_test - y_pred
        plt.scatter(y_test, residuals, color='green', label='Residuals')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f"{model} - Residual Plot")
        plt.xlabel('Actual Values')
        plt.ylabel('Residuals')
        plt.legend()

        # Convert plot to base64 for HTML rendering
        residuals_plot = io.BytesIO()
        plt.savefig(residuals_plot, format='png')
        residuals_plot.seek(0)
        residuals_plot_url = base64.b64encode(residuals_plot.getvalue()).decode('utf-8')
        plt.close()

        return render_template(
            'regression_evaluation.html',
            model=model,
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2=r2,
            pred_vs_actual_url=pred_vs_actual_url,
            residuals_plot_url=residuals_plot_url
        )
    except Exception as e:
        return str(e)

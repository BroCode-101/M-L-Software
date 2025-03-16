from flask import Blueprint, render_template, redirect, url_for, request, current_app,session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = Blueprint('model', __name__ , template_folder='templates')

@model.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@model.route('/select_model',methods=['GET','POST'])
def select_model():
    if request.method == 'POST':
        ml_type = request.form['model']
        session['ml_type'] = ml_type
        if ml_type == 'Supervised Learning':
            return redirect(url_for('model.supervised_type'))
        elif ml_type == 'Unsupervised Learning':
            return redirect(url_for('model.unsupervised_type'))
    return render_template('ml_type.html')

@model.route('/supervised_type',methods=['GET','POST'])
def supervised_type():
    if request.method == 'POST':
        supervised_type= request.form['model']
        session['supervised_type']=supervised_type
        if supervised_type == 'classification':
            return redirect(url_for('model.classification_type'))
        elif supervised_type == 'regression':
            return redirect(url_for('model.regression_type'))
    return render_template("supervised_type.html")

@model.route('/classification_type',methods=['GET','POST'])
def classification_type():
    if request.method == 'POST':
        classification_type = request.form['model']
        session['classification_model'] = classification_type
        return redirect(url_for('model.classification_upload'))
    return render_template('classification_type.html')

@model.route('/classification_upload', methods=['GET', 'POST'])
def classification_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('model.classification_display', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('classification_upload.html')

@model.route('/classification_display/<filename>', methods=['GET', 'POST'])
def classification_display(filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        if data.isnull().values.any():
            return "Dataset contains missing values. Please handle missing values first"
        if not data.select_dtypes(include='object').columns.empty:

            return "Dataset contains categorical columns. Please encode categorical columns first"
        data_html = data.head(10).to_html(index=False)

        return render_template('classification_display.html', tables=[data_html], titles=data.columns.values)
    except Exception as e:
        return str(e)


@model.route('/classification_eval/<filename>', methods=['GET'])
def classification_eval(filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        model = session['classification_model']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "SVM": SVC(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(),
            "Logistic Regression": LogisticRegression()
        }

        selected_model = models[model]
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return render_template(
            'evaluation.html',
            model=model,
            accuracy=accuracy,
            classification_report=report,
            conf_matrix=conf_matrix
        )
    except Exception as e:
        return str(e)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import io
import base64

@model.route('/regression_type', methods=['GET', 'POST'])
def regression_type():
    if request.method == 'POST':
        regression_type = request.form['model']
        session['regression_model'] = regression_type
        return redirect(url_for('model.regression_upload'))
    return render_template('regression_type.html')

@model.route('/regression_upload', methods=['GET', 'POST'])
def regression_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('model.regression_display', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('regression_upload.html')

@model.route('/regression_display/<filename>', methods=['GET', 'POST'])
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

@model.route('/regression_eval/<filename>', methods=['GET'])
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








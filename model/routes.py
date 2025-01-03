from flask import Blueprint, render_template, redirect, url_for, request, current_app
import pandas as pd
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

@model.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('model.display_dataset', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('predict.html')

@model.route('/display_dataset/<filename>', methods=['GET', 'POST'])
def display_dataset(filename):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        data_html = data.head(10).to_html(index=False)
        return render_template('display.html', tables=[data_html], titles=data.columns.values)
    except Exception as e:
        return str(e)

@model.route('/model_selection/<filename>', methods=['GET', 'POST'])
def model_selection(filename):
    if request.method == 'POST':
        selected_model = request.form['model']
        return redirect(url_for('model.evaluate', filename=filename, model=selected_model))
    return render_template('model_selection.html', filename=filename)

@model.route('/evaluate/<filename>/<model>', methods=['GET'])
def evaluate(filename, model):
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    try:
        data = pd.read_csv(filepath)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

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

from . import classification
from flask import  render_template, redirect, url_for, request, current_app,session
import pandas as pd
import os
import numpy as np
import pickle 
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from . import classification


@classification.route('/classification_type',methods=['GET','POST'])
def classification_type():
    if request.method == 'POST':
        classification_type = request.form['model']
        session['classification_model'] = classification_type
        return redirect(url_for('classification.classification_upload'))
    return render_template('classification_type.html')

@classification.route('/classification_upload', methods=['GET', 'POST'])
def classification_upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            return redirect(url_for('classification.classification_display', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('classification_upload.html')

@classification.route('/classification_display/<filename>', methods=['GET', 'POST'])
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


@classification.route('/classification_eval/<filename>', methods=['GET'])
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

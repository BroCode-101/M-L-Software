from flask import blueprints,render_template, request, redirect, url_for, send_file
import pandas as pd
import os
from werkzeug.utils import secure_filename
from . import handle_nan

UPLOAD_FOLDER = 'uploads/handle_nan'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@handle_nan.route('/upload_nan', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return redirect(url_for('handle_nan.handle_nan_values', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('upload_nan.html')


@handle_nan.route('/handle_nan/<filename>', methods=['GET', 'POST'])
def handle_nan_values(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        return str(e)

    if request.method == 'POST':
        option = request.form['nan_option']


        if option == 'drop':
            data = data.dropna()


        elif option == 'replace_mode':
            for column in data.columns:
                if data[column].isnull().any():
                    mode_value = data[column].mode()[0]
                    data[column].fillna(mode_value, inplace=True)


        cleaned_filename = f"cleaned_{filename}"
        cleaned_filepath = os.path.join(UPLOAD_FOLDER, cleaned_filename)
        data.to_csv(cleaned_filepath, index=False)

        return redirect(url_for('handle_nan.nan_preview', filename=cleaned_filename))


    nan_stats = data.isnull().sum().to_dict()
    total_nan = data.isnull().sum().sum()

    return render_template('handle_nan.html', nan_stats=nan_stats, total_nan=total_nan, filename=filename)


@handle_nan.route('/nan_preview/<filename>', methods=['GET', 'POST'])
def nan_preview(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        data = pd.read_csv(filepath)
        data_html = data.head(10).to_html(index=False)

        if request.method == 'POST':
            return redirect(url_for('handle_nan.download_file', filename=filename))

        return render_template('nan_preview.html', tables=[data_html], titles=data.columns.values, filename=filename)

    except Exception as e:
        return str(e)


@handle_nan.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return str(e)

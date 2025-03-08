from flask import  render_template, redirect, url_for, request, send_file
from werkzeug.utils import secure_filename
import pandas as pd
import os
from . import remove_columns

UPLOAD_FOLDER = 'uploads/remove_columns'
PROCESSED_FOLDER = 'uploads/processed'


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@remove_columns.route('/upload_remove', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return redirect(url_for('remove_columns.select_columns', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('upload_columns_remover.html')

@remove_columns.route('/select_columns/<filename>', methods=['GET', 'POST'])
def select_columns(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return "Error: File not found."

    try:
        data = pd.read_csv(file_path)
        columns = data.columns.tolist()

        if request.method == 'POST':
            selected_columns = request.form.getlist('columns')
            data.drop(columns=selected_columns, inplace=True)
            
            processed_file_path = os.path.join(PROCESSED_FOLDER, f"processed_{filename}")
            data.to_csv(processed_file_path, index=False)

            return redirect(url_for('remove_columns.download_file', filename=f"processed_{filename}"))

        return render_template('select_columns.html', columns=columns, filename=filename)
    
    except Exception as e:
        return f"Error: {str(e)}"

@remove_columns.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Error: File not found."

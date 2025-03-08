from flask import blueprints, render_template, request, redirect, url_for,session,send_file
import pandas as pd
import os
from werkzeug.utils import secure_filename
from . import encode_categorical
from urllib.parse import urlencode, parse_qs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

UPLOAD_FOLDER = 'uploads/encode_categorical'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@encode_categorical.route('/upload_encode', methods=['GET', 'POST'])
def upload_encode():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            return redirect(url_for('encode_categorical.handle_categorical', filename=filename))
        else:
            return "Please upload a valid CSV file."
    return render_template('upload_encode.html')

@encode_categorical.route('/display_categorical/<filename>', methods=['GET', 'POST'])
def handle_categorical(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        data = pd.read_csv(file_path)
        columns = data.columns
        categorical_columns = [x for x in columns if data[x].dtype == 'object']
        categorical_df = data[categorical_columns]
        session['categorical_columns'] = categorical_columns
        data_html = categorical_df.head(10).to_html(index=False)
        if request.method == 'POST':
            return redirect(url_for('encode_categorical.select_encoding', filename=filename))
        return render_template('display_categorical.html', tables=data_html, titles=categorical_df.columns.values, filename=filename)
    except Exception as e:
        return f"Error: {str(e)}"

@encode_categorical.route('/select_encoding/<filename>', methods=['POST', 'GET'])
def select_encoding(filename):
    categorical_columns = session.get('categorical_columns')
    if request.method == 'POST':
        selected_option = {}
        for category in categorical_columns:
            sanitized_category = category.replace(' ', '_')
            selected_value = request.form.get(f'option-{sanitized_category}')
            if selected_value:
                selected_option[category] = selected_value
            session['selected_option'] = selected_option
        return redirect(
            url_for('encode_categorical.selected_encoding', filename=filename)
        )
    return render_template('select_encoding.html', categorical_columns=categorical_columns)

@encode_categorical.route('/selected_encoding/<filename>', methods=['POST', 'GET'])
def selected_encoding(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        df = pd.read_csv(file_path)

        selected_options = session.get('selected_option',{})
        
        for col, encoder_type in selected_options.items():
            if col in df.columns:
                
                if encoder_type == 'Label':
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col])
                    
                elif encoder_type == 'One-Hot':
                    encoder = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_df = pd.DataFrame(encoder.fit_transform(df[[col]]), columns=encoder.get_feature_names_out([col]))
                    df = df.drop(col, axis=1).join(encoded_df)
                   
                elif encoder_type == 'Ordinal':
                    encoder = OrdinalEncoder()
                    df[col] = encoder.fit_transform(df[[col]])
            else:
                return 
        
        df.to_csv(file_path, index=False)
        return redirect(url_for('encode_categorical.encoding_preview', filename=filename))
    except Exception as e:
        return f"Error: {str(e)}"

@encode_categorical.route('/encode_preview/<filename>', methods=['GET', 'POST'])
def encoding_preview(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    selected_options = session.get('selected_option',{})
    try:
        data = pd.read_csv(filepath)
        data_html = data.head(10).to_html(index=False)
        if request.method == 'POST':
            return redirect(url_for('encode_categorical.download_file', filename=filename))
        return render_template('encoding_preview.html', selected_options =selected_options,tables=[data_html], titles=data.columns.values, filename=filename)
    except Exception as e:
        return str(e)
@encode_categorical.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return str(e)




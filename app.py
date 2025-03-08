from flask import Flask, render_template, redirect, url_for, session
from auth.routes import auth
from model.routes import model
from services.handle_nan import handle_nan
from services.encode_categorical import encode_categorical
from services.remove_columns import remove_columns
import os
import re
import pandas as pd

app = Flask(__name__)

app.secret_key = 'admin123'
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.register_blueprint(auth, url_prefix='/auth')
app.register_blueprint(model, url_prefix='/model')
app.register_blueprint(handle_nan,url_prefix='/handle_nan')
app.register_blueprint(encode_categorical,url_prefix = '/encode_categorical')
app.register_blueprint(remove_columns, url_prefix='/remove_columns')

@app.route('/')
def home():
    return redirect(url_for('auth.login'))

@app.route('/predict')
def predict():
    return redirect(url_for('model.predict'))

@app.route('/handle_nan')
def handle_nan():
    return redirect(url_for('handle_nan.upload_file'))

@app.route('/encode_categorical')
def handle_categorical():
    return redirect(url_for('encode_categorical.upload_encode'))

@app.route('/remove_columns')
def remove_columns():
    return redirect(url_for("remove_columns.upload"))


@app.route('/clear-dataset-logs-history', methods=['GET','POST'])
def clear_logs():
    folder_path = app.config['UPLOAD_FOLDER']  #CAUTION COULD DELETE PROJECT FOLDER
    delete_files_in_folder(folder_path)
    return "Dataset logs cleared successfully!"


def delete_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  # Delete each file
            print(f"Deleted: {file_path}")

@app.route('/Activate_God_mode1')
def god_mode1():
    df = pd.read_csv('users.csv')
    data = df.to_html(index =False)
    return render_template('render.html', tables=data, titles=list(df.columns))

def strip_html(value):
    return re.sub(r'<[^>]*>', '', value)
app.jinja_env.filters['strip_html'] = strip_html

if __name__ == '__main__':
    app.run(debug=True)


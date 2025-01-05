from flask import Flask, render_template, redirect, url_for, session
from auth.routes import auth
from model.routes import model
from services.handle_nan import handle_nan
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

@app.route('/')
def home():
    return redirect(url_for('auth.login'))
@app.route('/predict')
def predict():
    return redirect(url_for('model.predict'))
@app.route('/handle_nan')
def handle_nan():
    return redirect(url_for('handle_nan.upload_file'))
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

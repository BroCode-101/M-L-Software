from flask import Blueprint, render_template, redirect, url_for, request, session
import pandas as pd
import os

auth = Blueprint('auth', __name__,template_folder='templates')

csv_file = 'users.csv'
if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=['id', 'username', 'email', 'password'])
    df.to_csv(csv_file, index=False)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']

        df = pd.read_csv(csv_file)
        df['username'] = df['username'].astype(str)
        df['password'] = df['password'].astype(str)

        account = df[(df['username'] == username) & (df['password'] == password)]

        if not account.empty:
            session['logged'] = True
            session['id'] = int(account.iloc[0]['id'])
            session['username'] = account.iloc[0]['username']
            msg = 'Logged in successfully'
            return redirect(url_for('model.dashboard'))
        else:
            msg = 'Invalid username or password'
    return render_template('login.html', msg=msg)

@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'email' in request.form and 'password' in request.form:
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        df = pd.read_csv(csv_file)

        if not df[(df['username'] == username) & (df['email'] == email)].empty:
            msg = 'User already exists!'
        else:
            new_user = pd.DataFrame([[len(df)+1, username, email, password]], columns=['id', 'username', 'email', 'password'])
            df = pd.concat([df, new_user], ignore_index=True)
            df.to_csv(csv_file, index=False)
            msg = 'You have successfully registered!'
        
        return render_template('login.html', msg=msg)
    
    return render_template('signup.html')

@auth.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('auth.login'))

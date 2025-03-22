from flask import Blueprint, render_template, redirect, url_for, request, current_app,session

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
            return redirect(url_for('classification'))
        elif supervised_type == 'regression':
            return redirect(url_for('regression'))
    return render_template("supervised_type.html")










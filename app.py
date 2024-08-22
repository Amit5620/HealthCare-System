from flask import Flask, render_template, request, redirect, url_for, session, send_file
from werkzeug.utils import secure_filename
import os
from utils import load_model_and_predict

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Simple login logic for demonstration
        if username == 'amit' and password == '12345':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        disease = request.form['disease-select']
        image = request.files['image-upload']
        
        if image:
            filename = secure_filename(image.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
            
            prediction, affected_img_path = load_model_and_predict(filepath, disease)
            
            return render_template('result.html', disease=disease, filename=filename, prediction=prediction, affected_img=affected_img_path)
    
    return render_template('predict.html')


@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(path, as_attachment=True)



from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from FinalSimulator import simulate
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallbacksecret')

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'tracefile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['tracefile']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_name = os.path.splitext(filename)[0] + '.mp4'
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_name)
            file.save(input_path)

            simulate(input_path, output_path)

            return redirect(url_for('download_page', filename=output_name))

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/download_page/<filename>')
def download_page(filename):
    return render_template('download.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)

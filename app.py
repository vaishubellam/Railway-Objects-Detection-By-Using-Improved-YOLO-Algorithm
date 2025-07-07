from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from ultralytics import YOLO
import os
import subprocess
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Folder for uploaded images and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Load the YOLO model
model = YOLO('best.pt')  # Path to your trained YOLO weights

app.secret_key = 'supersecretkey'

# In-memory user database (For demonstration purposes only)
users = {}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/category')
def category():
    return render_template('index.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))
        
        users[username] = password
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video formats
            result_video_path = process_video(file_path, filename)
            return render_template('result.html', 
                                   original_file=filename, 
                                   result_video=result_video_path)

        else:  # Image file
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            annotated_frame = results[0].plot()
            cv2.imwrite(result_img_path, annotated_frame)
            return render_template('result.html', 
                                   original_file=filename, 
                                   result_file=f"result_{filename}")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')



def process_video(input_path, filename):
    # OpenCV Video Capture
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Temporary file for processed frames
    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary OpenCV video
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Processing: Annotate the frame with detections
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg Encoding for Final Output
    ffmpeg_command = [
        "ffmpeg", "-y",  # Overwrite output if exists
        "-i", temp_output,   # Input temporary video
        "-c:v", "libx264",   # H.264 codec for encoding
        "-preset", "medium", # Encoding speed
        "-crf", "23",        # Quality (Lower is better)
        "-c:a", "aac",       # Audio codec
        "-strict", "experimental",
        output_path          # Final output video
    ]

    try:
        # Run FFmpeg to re-encode
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    # Remove temporary file
    os.remove(temp_output)

    return f"result_{filename}"  # Return relative path

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    app.run(debug=True, port=5003)

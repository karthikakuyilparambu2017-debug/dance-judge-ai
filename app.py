from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import os
import threading
import subprocess
import time

app = Flask(__name__)
socketio = SocketIO(app)

# ✅ Folder setup
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Store the latest video path
latest_video = None

# ✅ Homepage Route (set to home.html)
@app.route('/')
def index():
    return render_template('home.html')

# ✅ Upload Page Route (was index.html, now upload.html)
@app.route('/index')
def upload_page():
    return render_template('index.html')

# ✅ Login Route
@app.route('/login')
def login():
    return render_template('login.html')

# ✅ Signup Route
@app.route('/signin')
def signin():
    return render_template('signin.html')

# ✅ About Page
@app.route('/about')
def about():
    return render_template('about.html')

# ✅ Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# ✅ Explicit Home Page Route (optional direct access)
@app.route('/home')
def home():
    return render_template('home.html')

# ✅ Upload Route (AJAX request)
@app.route('/upload', methods=['POST'])
def upload():
    global latest_video

    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # ✅ Create a unique filename
    filename = f"{int(time.time())}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # ✅ Save the uploaded file
    file.save(filepath)
    latest_video = filepath

    # ✅ Start Dance Judge in background
    threading.Thread(target=run_dance_judge, args=(filepath,)).start()

    return jsonify({'video_url': f'/static/uploads/{filename}', 'message': 'Video uploaded successfully'}), 200

# ✅ Run the Dance Judge AI script
def run_dance_judge(filepath):
    global latest_video
    if filepath != latest_video:
        return

    try:
        command = f"python dance_judge.py \"{filepath}\""
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in process.stdout:
            line = line.strip()
            print(f"Output: {line}")
            if "Score" in line:
                socketio.emit('score_update', {'message': line})

        process.wait()

    except Exception as e:
        print(f"Error: {e}")
        socketio.emit('error', {'error': str(e)})

# ✅ Serve uploaded videos
@app.route('/static/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ✅ SocketIO Connection
@socketio.on('connect')
def on_connect():
    print("Client connected")

# ✅ Run App
if __name__ == '__main__':
    socketio.run(app, debug=True)

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
import tempfile
import base64
from werkzeug.utils import secure_filename
from PIL import Image

# --- Configuration ---
app = Flask(__name__)
# Set a secret key for session management (needed for flash messages)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super_secret_dev_key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}
app.config['OUTPUT_FOLDER'] = 'static/output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define Cascade file paths (User MUST place these in the project root)
# The user must download 'cars.xml' and 'bus.xml' and place them here.
CAR_CASCADE_PATH = r'cascade_files\cars.xml'
BUS_CASCADE_PATH = r'cascade_files\Bus_front.xml' 

# Load Cascade Classifiers globally
# Note: The app will run, but processing will fail if the XML files are missing/corrupted.
car_cascade = cv2.CascadeClassifier(CAR_CASCADE_PATH)
bus_cascade = cv2.CascadeClassifier(BUS_CASCADE_PATH)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(file_path):
    """Processes an image file for vehicle detection and returns base64 string."""
    try:
        # Check if cascades were loaded successfully
        if car_cascade.empty() or bus_cascade.empty():
            return "ERROR: Cascade XML files not found. Please check 'cars.xml' and 'bus.xml' paths in the project root.", None

        # Read the image
        img = cv2.imread(file_path)
        
        # Resize for consistent processing (matching notebook style of 450x250)
        img = cv2.resize(img, (450, 250))
        img_copy = img.copy()

        # Preprocessing steps from the notebook
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        dilated = cv2.dilate(blur, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # 1. Detect Cars 
        cars = car_cascade.detectMultiScale(closing, 1.1, 1)
        
        car_count = 0
        for (x, y, w, h) in cars:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red rectangle for cars
            car_count += 1

        # 2. Detect Buses 
        buses = bus_cascade.detectMultiScale(gray, 1.1, 1)

        bus_count = 0
        for (x, y, w, h) in buses:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2) # Blue rectangle for buses
            bus_count += 1
            
        # Add text overlay for count
        total_count = car_count + bus_count
        cv2.putText(img_copy, f'Cars: {car_count}, Buses: {bus_count}, Total: {total_count}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


        # Encode processed image to JPEG for web display
        _, buffer = cv2.imencode('.jpeg', img_copy)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        message = f"Detection complete. Found {car_count} cars and {bus_count} buses in the image."
        return message, img_base64

    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return f"An error occurred during image processing: {e}", None

def process_video(file_path, output_filename):
    """Processes a video file for vehicle detection and saves the output."""
    try:
        if car_cascade.empty():
            return "ERROR: Car Cascade XML file not found. Video processing aborted."

        cap = cv2.VideoCapture(file_path)
        
        if not cap.isOpened():
            return "Error opening video file."

        # Define the codec and create VideoWriter object
        # Target size matching notebook
        target_width, target_height = 450, 250
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Output video path (using .avi and XVID codec as per notebook)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            frame = cv2.resize(frame, (target_width, target_height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use car cascade for detection
            cars = car_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            # Draw rectangles
            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green rectangle

            # Simple count visualization
            car_count = len(cars)
            cv2.putText(frame, f'Detected: {car_count} vehicles', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        
        return f"Video processing complete. {frame_count} frames processed.", output_path.replace('static/', '')

    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        return f"An error occurred during video processing: {e}", None
    
@app.route('/', methods=['GET'])
def index():
    """Renders the main upload page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and processing."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # Use a temporary file path for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        try:
            if file_ext in {'png', 'jpg', 'jpeg'}:
                # Process image
                message, result_data = process_image(temp_path)
                
                if result_data:
                    return render_template('index.html', 
                                           message=message, 
                                           result_image=result_data,
                                           is_video=False)
                else:
                    flash(message)
                    return redirect(url_for('index'))
                    
            elif file_ext in {'mp4', 'avi'}:
                # Process video
                output_filename = 'output_' + os.path.splitext(filename)[0] + '.avi' 
                message, result_path = process_video(temp_path, output_filename)
                
                if result_path:
                    flash("Note: The output video is in AVI format (.avi) as per the source notebook. AVI is not universally supported in browsers. If the video doesn't play, you might need to convert the file externally.")
                    return render_template('index.html', 
                                           message=message, 
                                           result_video=result_path,
                                           is_video=True)
                else:
                    flash(message)
                    return redirect(url_for('index'))

        finally:
            # Clean up the temporary file
            os.unlink(temp_path)
            
    else:
        flash('File type not allowed. Please upload an image (jpg, png) or video (mp4, avi).')
        return redirect(url_for('index'))

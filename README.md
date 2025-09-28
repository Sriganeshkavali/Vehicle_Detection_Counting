# 🚗 Vehicle Detection and Counting Web App
This is a Flask-based web application for detecting and counting vehicles in uploaded images and videos using OpenCV and pre-trained Haar Cascade classifiers, based on your Jupyter Notebook logic.

# ⚙️ Prerequisites
You need Python 3.8+ installed on your system.

1. Install Dependencies
You will need the following Python libraries. Create a file named requirements.txt:

Flask
opencv-python
numpy
werkzeug
Pillow

Install them using pip:

pip install -r requirements.txt

2. Obtain Haar Cascade XML Files (Crucial Step)
The core detection logic relies on the pre-trained Haar Cascade XML files. The application is configured to look for them in the root directory.

You must download the required XML files and place them in the root directory of this project (next to app.py).

cars.xml (for general car detection)

bus.xml (or potentially Bus_front.xml)

Recommendation: Search online for the files, e.g., "OpenCV Haar Cascade cars xml". The official OpenCV GitHub repository is a great place to start. If you only use one cascade for simplicity, ensure you rename it to cars.xml or update the path in app.py.

3. Project Structure
Ensure your project directory looks like this:

vehicle-detection-app/
├── app.py
├── cars.xml        <-- PLACE CARS XML HERE
├── bus.xml         <-- PLACE BUS XML HERE
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    └── output/     <-- (Created automatically, processed videos are saved here)

# 🚀 How to Run the App
Set the Secret Key: Flask requires a secret key for security.

# On Linux/macOS
export FLASK_SECRET_KEY='your_strong_secret_key'

# On Windows (Command Prompt)
set FLASK_SECRET_KEY=your_strong_secret_key

(For quick local testing, the default key is used if the environment variable is not set.)

Run the Flask Application:

python app.py

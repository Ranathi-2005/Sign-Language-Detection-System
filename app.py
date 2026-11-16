from flask import Flask, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and label binarizer
model = load_model('sign_language_cnn_model.h5')

# Labels for sign language prediction
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y","Z"]

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to get the camera feed
def get_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera index {i}")
            return cap
    print("❌ No accessible camera found.")
    return None

camera = get_camera()

# Function to process and predict the hand gesture
def predict_from_hand_region(hand_img):
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28)) / 255.0
    reshaped = resized.reshape(1, 28, 28, 1)
    pred = model.predict(reshaped)
    return labels[np.argmax(pred)]

# Generate frames for video streaming
def gen_frames():
    if camera is None:
        raise RuntimeError("❌ Camera not found.")
    while True:
        success, frame = camera.read()
        if not success:
            print("❌ Failed to read frame")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        label = "No Hand Detected"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get bounding box around hand
                h, w, _ = frame.shape
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                # Convert normalized coordinates to pixel values
                x1, y1 = int(x_min * w), int(y_min * h)
                x2, y2 = int(x_max * w), int(y_max * h)
                
                # Add margin
                x1, y1 = max(x1 - 20, 0), max(y1 - 20, 0)
                x2, y2 = min(x2 + 20, w), min(y2 + 20, h)

                hand_img = frame[y1:y2, x1:x2]
                if hand_img.size > 0:
                    label = predict_from_hand_region(hand_img)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                break  # Use first hand only

        cv2.putText(frame, f'Prediction: {label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)







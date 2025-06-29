from flask import Flask, Response
from ultralytics import YOLO
import numpy as np
from keras.models import load_model
from keras.applications import InceptionV3
from keras.preprocessing import image as keras_image
import keras
import cv2
import math
import threading
from queue import Queue


app = Flask(__name__)

model_save_path = "C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/video_classifier_model_lstm.h5"

sequence_model = load_model(model_save_path)
print("Model loaded successfully")

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100
MAX_SEQ_LENGTH = 7
NUM_FEATURES = 2048

# Define the feature extractor
def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

# Initialize the feature extractor
feature_extractor = build_feature_extractor()

# Global variable to hold frames
frames = []

# Function to preprocess frames for model prediction
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0
    return frame

# Function to prepare a single video for sequence prediction
def prepare_single_video(frames):
    frames = np.array(frames)
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i in range(min(MAX_SEQ_LENGTH, len(frames))):
        processed_frame = preprocess_frame(frames[i])
        processed_frame = np.expand_dims(processed_frame, axis=0)
        frame_features[0, i, :] = feature_extractor.predict(processed_frame)
        frame_mask[0, i] = 1

    return frame_features, frame_mask

# Define a queue for communication between threads
prediction_queue = Queue()
last_prob=None

# Function to perform sequence prediction on frames
def sequence_prediction_on_frames(frames):
   
    global last_prob
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]
    last_prob=probabilities
    print('$$$$$$$$$$$$$$$$$$$$$$************$$$$$$$$$$$$$$$$$$$$$$$$$4')
    

# Function to generate live video feed
# Function to generate live video feed
def generate_feed():
    global frames

    cap = cv2.VideoCapture("C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/FIGHT_PRACTICE.mp4")


    cap.set(3, 640)
    cap.set(4, 480)

    model_paths = [
        "C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/detect/train/weights/best.pt",  
        "C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/detect/weapondetction1_train/weights/best.pt",
        "C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/detect/weapondetction1_train/weights/best.pt",
        "C:/Users/amank/Desktop/thread_detection_PR/EyeonGithub/detect/fire_smoke_train/weights/best.pt"
    ]

    models = [YOLO(path) for path in model_paths]

    classNames_list = [
        ["masked", "person", "masked"],
        ["weapon"],
        ["weapon"],
        ["fire", "smoke"]
    ]

    while True:
        success, frame = cap.read()

        if not success:
            break

        frames.append(frame)

        if len(frames) == MAX_SEQ_LENGTH:
            threading.Thread(target=sequence_prediction_on_frames, args=(frames,)).start()
            frames = []

        for i, (model, classNames) in enumerate(zip(models, classNames_list)):
            results = model(frame, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    if confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        cls = int(box.cls[0])
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        thickness = 2

                        if classNames[cls] == "masked" and confidence > 0.7:
                            color = (255, 0, 255)
                        elif classNames[cls] == "person":
                            color = (255, 0, 0)
                        elif classNames[cls] == "weapon":
                            color = (45, 4, 210)
                        elif classNames[cls] == "fire":
                            color = (0, 0, 255)
                        else:
                            continue

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, classNames[cls], org, font, fontScale, color, thickness)

        if last_prob is not None:
            class_names = ["Non_violence", "Violence"]
            top_class_index = np.argmax(last_prob)
            top_class_label = class_names[top_class_index]
            top_probability = last_prob[top_class_index]

            cv2.putText(frame, f"Prediction: {top_class_label} ({top_probability:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
()

# API endpoint to serve live images
@app.route('/live_feed')
def live_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

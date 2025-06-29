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
from collections import deque


app = Flask(__name__)
   

# Function to generate live video feed
def generate_feed():
    print("Loading model ...")
    model_vio = load_model("./detect/modelnew.h5")
    Q = deque(maxlen=128)

    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    model_paths = [
        'detect/train/weights/best.pt',
        'detect/weapondetction1_train/weights/best.pt',
        'detect/weapondetction1_train/weights/best.pt',
        'detect/fire_smoke_train/weights/best.pt'
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
        input = frame.copy()

        # Preprocess the frame
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, (128, 128)).astype("float32") / 255

        # Make predictions on the frame
        preds = model_vio.predict(np.expand_dims(input, axis=0))[0]
        Q.append(preds)

        # Perform prediction averaging over the current history of previous predictions
        results = np.array(Q).mean(axis=0)
        label = 1 if results > 0.60 else 0
      
        
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

                        if classNames[cls] == "masked" and confidence>0.7 :
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
        text_color = (0, 255, 0)  # Default: green
        if label:  # Violence prediction
            text_color = (0, 0, 255)  # Red

        text = "Violence: {}".format(label)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (35, 50), FONT, 1.25, text_color, 3)
        
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        

    cap.release()

# API endpoint to serve live images
@app.route('/live_feed')
def live_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

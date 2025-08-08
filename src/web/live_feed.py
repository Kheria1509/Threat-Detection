from flask import Flask, Response
from ultralytics import YOLO
import cv2
import math

app = Flask(__name__)

# Function to generate live video feed
def generate_feed():
    # Open a video capture object
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

    # Class names for each model
    classNames_list = [
        ["masked", "person", "masked"],  # Update with the correct class names for model 1
        ["weapon"],  # Update with the correct class names for model 2
        ["weapon"],  # Update with the correct class names for model 3
        ["fire", "smoke"]   # Update with the correct class names for model 4
    ]

    total_predictions = [0] * len(models)
    total_correct = [0] * len(models)

    while True:
        success, img = cap.read()

        if not success:
            break

        # Read a frame from the camera
        for i, (model, classNames) in enumerate(zip(models, classNames_list)):
            results = model(img, stream=True)

            # Coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # Confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    # Check if confidence is greater than 0.5
                    if confidence > 0.4:
                        # Bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                        # Class name
                        cls = int(box.cls[0])

                        # Object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        thickness = 2

                        # Set the rectangle color based on class label
                        if classNames[cls] == "masked":
                            color = (255, 0, 255)  # Pink for masked person
                        elif classNames[cls] == "person":
                            color = (255, 0, 0)  # Blue for person
                        elif classNames[cls] == "weapon":
                            color = (45, 4, 210)  # Red for weapon
                        elif classNames[cls] == "fire":
                            color = (0, 0, 255)  # Red for fire
                        else:
                            continue  # Skip drawing rectangles for other classes

                        total_predictions[i] += 1  # Increment total predictions count

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        
        if not ret:
            continue
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# API endpoint to serve live images
@app.route('/live_feed')
def live_feed():
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

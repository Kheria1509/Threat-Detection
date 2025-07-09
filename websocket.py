import cv2
import json
import asyncio
import websockets
import base64
from ultralytics import YOLO

# Load models
weapon_model = YOLO(r'M:/mask/detect/threat_train/weights/best.pt')
fire_smoke_model = YOLO(r'detect/fire_smoke_train/weights/best.pt')

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Function to detect objects in a frame
def detect_objects(frame):
    results = []

    # Detect weapons
    weapon_results = weapon_model(frame, stream=True)
    for r in weapon_results:
        for box in r.boxes:
            confidence = float(box.conf[0])  # Convert tensor to float
            cls = int(box.cls[0])
            class_name = weapon_class_names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list and then to int

            if confidence > 0.50:  # Threshold for weapon detection
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 255)  # Red for weapons
                })

    # Detect fire and smoke
    fire_results = fire_smoke_model(frame, stream=True)
    for r in fire_results:
        for box in r.boxes:
            confidence = float(box.conf[0])  # Convert tensor to float
            cls = int(box.cls[0])
            class_name = fire_smoke_class_names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list and then to int

            if class_name == "fire" and confidence > 0.50:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 165, 255)  # Orange for fire
                })
            elif class_name == "smoke" and confidence > 0.90:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 0)  # Black for smoke
                })

    return results

# WebSocket server function
async def video_stream(websocket, path):
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Detect objects
            detections = detect_objects(frame)

            # Draw boxes and labels on the frame
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                color = detection["color"]
                label = f"{detection['class']} {detection['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # Prepare the data to send
            data = {
                "detections": detections,
                "frame": frame_encoded
            }

            # Send data to the client
            await websocket.send(json.dumps(data))

            # Limit the frame rate
            await asyncio.sleep(0.033)  # Roughly 30 FPS

    finally:
        cap.release()

# Start the WebSocket server
start_server = websockets.serve(video_stream, "localhost", 8765)

# Run the server
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()

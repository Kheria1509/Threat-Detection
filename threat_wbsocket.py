import cv2
import json
import asyncio
import websockets
import base64
from ultralytics import YOLO
from collections import deque
from threading import Thread
import time
import winsound  # Use for playing sound on Windows

# Load models
weapon_model = YOLO(r'M:/mask/detect/threat_train/weights/best.pt')
fire_smoke_model = YOLO(r'detect/fire_smoke_train/weights/best.pt')

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Store clients and threat status
clients = {
    "local_master": None,
    "regional_master": None
}
threat_detected = False
threat_acknowledged = False
threat_detection_window = deque(maxlen=30)  # Store last 30 frames for threat analysis

# Define thresholds
THREAT_CONFIDENCE_THRESHOLD = 0.50
FIRE_CONFIDENCE_THRESHOLD = 0.50
SMOKE_CONFIDENCE_THRESHOLD = 0.90
ALARM_FILE = 'alarm.wav'  # Replace with path to your alarm sound file

# Function to play an alarm sound
def play_alarm():
    winsound.Beep(1000, 1000)  # Frequency, duration in milliseconds
    # winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME)

# Define threat detection logic
def is_threat(detections):
    global threat_detection_window
    threat_detection_window.append(detections)
    
    if len(threat_detection_window) < threat_detection_window.maxlen:
        return False  # Not enough data to make a decision
    
    threat_count = 0
    for frame_detections in threat_detection_window:
        if any(d['class'] in ["gun", "knife", "fire", "smoke"] and d['confidence'] > THREAT_CONFIDENCE_THRESHOLD for d in frame_detections):
            threat_count += 1
    
    return threat_count > (threat_detection_window.maxlen / 2)

def detect_objects(frame):
    results = []

    # Detect weapons
    weapon_results = weapon_model(frame, stream=True)
    for r in weapon_results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = weapon_class_names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if confidence > THREAT_CONFIDENCE_THRESHOLD:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 255)
                })

    # Detect fire and smoke
    fire_results = fire_smoke_model(frame, stream=True)
    for r in fire_results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = fire_smoke_class_names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if class_name == "fire" and confidence > FIRE_CONFIDENCE_THRESHOLD:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 165, 255)
                })
            elif class_name == "smoke" and confidence > SMOKE_CONFIDENCE_THRESHOLD:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 0)
                })

    return results

# WebSocket server function
async def video_stream(websocket, path):
    global threat_detected, threat_acknowledged
    user_type = path.strip('/')
    if user_type not in ["local_master", "regional_master"]:
        await websocket.close()
        return

    clients[user_type] = websocket
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            detections = detect_objects(frame)
            threat_detected = is_threat(detections)

            # Draw boxes and labels
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                color = detection["color"]
                label = f"{detection['class']} {detection['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_encoded = base64.b64encode(buffer).decode('utf-8')

            # Prepare data to send
            data = {
                "detections": detections,
                "frame": frame_encoded,
                "threat_detected": threat_detected
            }

            # Send data to the local master
            if clients["local_master"]:
                await clients["local_master"].send(json.dumps(data))

                if threat_detected and not threat_acknowledged:
                    play_alarm()  # Play alarm when a threat is detected
                    # Start a timeout thread to notify regional master if not acknowledged
                    Thread(target=threat_timeout).start()

            # Notify regional master
            if clients["regional_master"]:
                notification = json.dumps({
                    "message": "Threat detected",
                    "threat_detected": threat_detected
                })
                await clients["regional_master"].send(notification)

            await asyncio.sleep(0.033)

    finally:
        cap.release()
        clients[user_type] = None

async def threat_timeout():
    global threat_detected, threat_acknowledged

    # Wait for 10 seconds for acknowledgment
    time.sleep(10)

    if threat_detected and not threat_acknowledged:
        # Notify regional master and play alarm
        if clients["regional_master"]:
            notification = json.dumps({"message": "Threat not acknowledged", "threat_detected": threat_detected})
            await clients["regional_master"].send(notification)
            play_alarm()

async def handle_acknowledgment(websocket, path):
    global threat_acknowledged
    if path.strip('/') == "acknowledge_threat":
        await websocket.recv()
        threat_acknowledged = True
        # Notify regional master about threat acknowledgment
        if clients["regional_master"]:
            await clients["regional_master"].send(json.dumps({"message": "Threat acknowledged"}))

# Start the WebSocket servers
start_server = websockets.serve(video_stream, "localhost", 8765)
acknowledge_server = websockets.serve(handle_acknowledgment, "localhost", 8766)

# Run the servers
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_until_complete(acknowledge_server)
asyncio.get_event_loop().run_forever()
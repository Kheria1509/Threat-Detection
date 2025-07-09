import cv2
import json
import asyncio
import websockets
import base64
import smtplib
from email.mime.text import MIMEText
from collections import defaultdict
from datetime import datetime, timedelta
from ultralytics import YOLO
import pygame

# Initialize pygame mixer for playing sounds
pygame.mixer.init()

# Load models
weapon_model = YOLO(r'M:/mask/detect/threat_train/weights/best.pt')
fire_smoke_model = YOLO(r'detect/fire_smoke_train/weights/best.pt')

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Thresholds and limits
CONTINUOUS_DETECTION_THRESHOLD = 10
THREAT_ACKNOWLEDGMENT_TIMEOUT = 10  # seconds
THREAT_BURST_LIMIT = 3
THREAT_BURST_TIMEFRAME = timedelta(minutes=5)

# Detection counters and threat history
detection_counters = defaultdict(int)
threat_history = []
acknowledgment_timers = {}

# Master connections
local_master = None
global_master = None

# Email configuration
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_email_password"
EMAIL_RECEIVER = "alert_receiver@example.com"

# Function to send an email alert
def send_email_alert(threat_type):
    msg = MIMEText(f"A {threat_type} threat has been detected and not acknowledged.")
    msg['Subject'] = "Threat Alert"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP_SSL('smtp.example.com', 465)  # Use your SMTP server details
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to detect objects in a frame
def detect_objects(frame):
    results = []
    detected_threats = set()

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
                detected_threats.add(class_name)

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
                detected_threats.add(class_name)
            elif class_name == "smoke" and confidence > 0.90:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 0)  # Black for smoke
                })
                detected_threats.add(class_name)

    return results, detected_threats

# WebSocket server function
async def video_stream(websocket, path):
    global local_master, global_master

    if path == "/local_master":
        local_master = websocket
    elif path == "/global_master":
        global_master = websocket

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
            detections, detected_threats = detect_objects(frame)

            # Update detection counters
            for threat in detected_threats:
                detection_counters[threat] += 1
                if detection_counters[threat] >= CONTINUOUS_DETECTION_THRESHOLD:
                    threat_time = datetime.now()
                    threat_history.append((threat, threat_time))
                    # Check for threat bursts
                    recent_threats = [
                        t for t, time in threat_history
                        if time > threat_time - THREAT_BURST_TIMEFRAME
                    ]
                    if len(recent_threats) >= THREAT_BURST_LIMIT:
                        await alert_user("global", threat)
                    else:
                        await alert_user("local", threat)
                    detection_counters[threat] = 0  # Reset counter after alert

            # Reset counters for non-detected threats
            for threat in list(detection_counters.keys()):
                if threat not in detected_threats:
                    detection_counters[threat] = 0

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

            # Send data to connected users
            if local_master and path == "/local_master":
                await local_master.send(json.dumps(data))
            if global_master and path == "/global_master":
                await global_master.send(json.dumps(data))

            # Limit the frame rate
            await asyncio.sleep(0.033)  # Roughly 30 FPS

    finally:
        cap.release()

# Function to alert the user
async def alert_user(user_type, threat):
    if user_type == "local" and local_master:
        acknowledgment_timers[threat] = asyncio.get_event_loop().call_later(
            THREAT_ACKNOWLEDGMENT_TIMEOUT, handle_timeout, "local", threat
        )
        await send_alert(local_master, threat)
    elif user_type == "global" and global_master:
        acknowledgment_timers[threat] = asyncio.get_event_loop().call_later(
            THREAT_ACKNOWLEDGMENT_TIMEOUT, handle_timeout, "global", threat
        )
        await send_alert(global_master, threat)
    else:
        send_email_alert(threat)

# Function to send alert to a user
async def send_alert(websocket, threat):
    alert_message = {
        "alert": f"{threat} threat detected",
        "acknowledge": True,
        "play_alarm": True  # Send command to play alarm sound on client side
    }
    await websocket.send(json.dumps(alert_message))
    play_alarm_sound()  # Play alarm locally as well

# Function to handle timeout
def handle_timeout(user_type, threat):
    if user_type == "local":
        if global_master:
            asyncio.create_task(alert_user("global", threat))
        else:
            send_email_alert(threat)
    elif user_type == "global":
        send_email_alert(threat)

# Function to play an alarm sound
def play_alarm_sound():
    try:
        pygame.mixer.music.load('alarm_sound.mp3')  # Replace with your alarm sound file
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play alarm sound: {e}")

# Start the WebSocket server
async def main():
    async with websockets.serve(video_stream, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Run the server
asyncio.run(main())
import cv2
import json
import asyncio
import websockets
import base64
import smtplib
from email.mime.text import MIMEText
from collections import defaultdict
from datetime import datetime, timedelta
from ultralytics import YOLO
import pygame

# Initialize pygame mixer for playing sounds
pygame.mixer.init()

# Load models
weapon_model = YOLO(r'M:/mask/detect/threat_train/weights/best.pt')
fire_smoke_model = YOLO(r'detect/fire_smoke_train/weights/best.pt')

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Thresholds and limits
CONTINUOUS_DETECTION_THRESHOLD = 10
THREAT_ACKNOWLEDGMENT_TIMEOUT = 10  # seconds
THREAT_BURST_LIMIT = 3
THREAT_BURST_TIMEFRAME = timedelta(minutes=5)

# Detection counters and threat history
detection_counters = defaultdict(int)
threat_history = []
acknowledgment_timers = {}

# Master connections
local_master = None
global_master = None

# Email configuration
EMAIL_SENDER = "your_email@example.com"
EMAIL_PASSWORD = "your_email_password"
EMAIL_RECEIVER = "alert_receiver@example.com"

# Function to send an email alert
def send_email_alert(threat_type):
    msg = MIMEText(f"A {threat_type} threat has been detected and not acknowledged.")
    msg['Subject'] = "Threat Alert"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER

    try:
        server = smtplib.SMTP_SSL('smtp.example.com', 465)  # Use your SMTP server details
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to detect objects in a frame
def detect_objects(frame):
    results = []
    detected_threats = set()

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
                detected_threats.add(class_name)

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
                detected_threats.add(class_name)
            elif class_name == "smoke" and confidence > 0.90:
                results.append({
                    "class": class_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                    "color": (0, 0, 0)  # Black for smoke
                })
                detected_threats.add(class_name)

    return results, detected_threats

# WebSocket server function
async def video_stream(websocket, path):
    global local_master, global_master

    if path == "/local_master":
        local_master = websocket
    elif path == "/global_master":
        global_master = websocket

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
            detections, detected_threats = detect_objects(frame)

            # Update detection counters
            for threat in detected_threats:
                detection_counters[threat] += 1
                if detection_counters[threat] >= CONTINUOUS_DETECTION_THRESHOLD:
                    threat_time = datetime.now()
                    threat_history.append((threat, threat_time))
                    # Check for threat bursts
                    recent_threats = [
                        t for t, time in threat_history
                        if time > threat_time - THREAT_BURST_TIMEFRAME
                    ]
                    if len(recent_threats) >= THREAT_BURST_LIMIT:
                        await alert_user("global", threat)
                    else:
                        await alert_user("local", threat)
                    detection_counters[threat] = 0  # Reset counter after alert

            # Reset counters for non-detected threats
            for threat in list(detection_counters.keys()):
                if threat not in detected_threats:
                    detection_counters[threat] = 0

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

            # Send data to connected users
            if local_master and path == "/local_master":
                await local_master.send(json.dumps(data))
            if global_master and path == "/global_master":
                await global_master.send(json.dumps(data))

            # Limit the frame rate
            await asyncio.sleep(0.033)  # Roughly 30 FPS

    finally:
        cap.release()

# Function to alert the user
async def alert_user(user_type, threat):
    if user_type == "local" and local_master:
        acknowledgment_timers[threat] = asyncio.get_event_loop().call_later(
            THREAT_ACKNOWLEDGMENT_TIMEOUT, handle_timeout, "local", threat
        )
        await send_alert(local_master, threat)
    elif user_type == "global" and global_master:
        acknowledgment_timers[threat] = asyncio.get_event_loop().call_later(
            THREAT_ACKNOWLEDGMENT_TIMEOUT, handle_timeout, "global", threat
        )
        await send_alert(global_master, threat)
    else:
        send_email_alert(threat)

# Function to send alert to a user
async def send_alert(websocket, threat):
    alert_message = {
        "alert": f"{threat} threat detected",
        "acknowledge": True,
        "play_alarm": True  # Send command to play alarm sound on client side
    }
    await websocket.send(json.dumps(alert_message))
    play_alarm_sound()  # Play alarm locally as well

# Function to handle timeout
def handle_timeout(user_type, threat):
    if user_type == "local":
        if global_master:
            asyncio.create_task(alert_user("global", threat))
        else:
            send_email_alert(threat)
    elif user_type == "global":
        send_email_alert(threat)

# Function to play an alarm sound
def play_alarm_sound():
    try:
        pygame.mixer.music.load('alarm_sound.mp3')  # Replace with your alarm sound file
        pygame.mixer.music.play()
    except Exception as e:
        print(f"Failed to play alarm sound: {e}")

# Start the WebSocket server
async def main():
    async with websockets.serve(video_stream, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Run the server
asyncio.run(main())

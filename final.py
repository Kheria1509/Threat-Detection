# Fixed backend code with proper WebSocket handling
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
weapon_model = YOLO(r'detect\threat_train\weights\best.pt')
fire_smoke_model = YOLO(r'detect/fire_smoke_train/weights/best.pt')

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Store clients and threat status
clients = {
    "local_master": set(),  # Changed to set to handle multiple connections
    "regional_master": set()
}
threat_detected = False
threat_acknowledged = False
threat_detection_window = deque(maxlen=30)

# Define thresholds
THREAT_CONFIDENCE_THRESHOLD = 0.50
FIRE_CONFIDENCE_THRESHOLD = 0.50
SMOKE_CONFIDENCE_THRESHOLD = 0.90
ALARM_FILE = 'alarm.wav'

# Function to play an alarm sound
def play_alarm():
    print("Playing alarm")
    # winsound.Beep(1000, 1000)
    # winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME)

def stop_alarm():
    # winsound.PlaySound(None, winsound.SND_PURGE)
    print("Stopping alarm")

# Define threat detection logic
def is_threat(detections):
    global threat_detection_window, threat_detected
    threat_detection_window.append(detections)
    
    if len(threat_detection_window) < threat_detection_window.maxlen:
        return False
    
    if threat_detected:
        return True
    
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

# Unified WebSocket handler for all connections
async def websocket_handler(websocket, path):
    global threat_detected, threat_acknowledged
    
    print(f"New connection from path: {path}")
    
    try:
        # Handle video streaming connections
        if path in ["/local_master", "/regional_master"]:
            user_type = path.strip('/')
            clients[user_type].add(websocket)
            
            if user_type == "local_master":
                await handle_video_stream(websocket)
            else:  # regional_master
                await handle_regional_master(websocket)
                
        # Handle acknowledgment connections
        elif path == "/acknowledge":
            await handle_acknowledgment(websocket)
        else:
            print(f"Unknown path: {path}")
            await websocket.close()
            
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection closed for path: {path}")
    except Exception as e:
        print(f"Error in websocket_handler: {e}")
    finally:
        # Clean up client from all sets
        for client_set in clients.values():
            client_set.discard(websocket)

async def handle_video_stream(websocket):
    global threat_detected, threat_acknowledged
    
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

            # Send data to the client
            await websocket.send(json.dumps(data))

            if threat_detected and not threat_acknowledged:
                play_alarm()
                # Start a timeout thread to notify regional master if not acknowledged
                asyncio.create_task(threat_timeout())

            await asyncio.sleep(0.033)

    finally:
        cap.release()

async def handle_regional_master(websocket):
    """Handle regional master connections - just keep connection alive and send notifications"""
    try:
        while True:
            if threat_detected:
                notification = json.dumps({
                    "message": "Threat detected",
                    "threat_detected": threat_detected
                })
                await websocket.send(notification)
            await asyncio.sleep(1)
    except websockets.exceptions.ConnectionClosed:
        pass

async def handle_acknowledgment(websocket):
    global threat_acknowledged, threat_detected
    
    print("Acknowledgment handler started")
    
    try:
        async for message in websocket:
            print(f"Received acknowledgment message: {message}")
            data = json.loads(message)
            
            if data.get('action') == 'acknowledge_threat':
                threat_acknowledged = True
                threat_detected = False  # Reset threat status
                stop_alarm()
                print("Threat acknowledged and reset")

                # Notify all regional masters about threat acknowledgment
                regional_clients = list(clients["regional_master"])
                for client in regional_clients:
                    try:
                        await client.send(json.dumps({"message": "Threat acknowledged"}))
                    except:
                        clients["regional_master"].discard(client)
                        
                # Send confirmation back to the client
                await websocket.send(json.dumps({"status": "acknowledged"}))
                
    except websockets.exceptions.ConnectionClosed:
        print("Acknowledgment connection closed")
    except Exception as e:
        print(f"Error in handle_acknowledgment: {e}")

async def threat_timeout():
    global threat_detected, threat_acknowledged

    # Wait for 10 seconds for acknowledgment
    await asyncio.sleep(10)

    if threat_detected and not threat_acknowledged:
        # Notify regional masters
        regional_clients = list(clients["regional_master"])
        for client in regional_clients:
            try:
                notification = json.dumps({
                    "message": "Threat not acknowledged", 
                    "threat_detected": threat_detected
                })
                await client.send(notification)
                play_alarm()
            except:
                clients["regional_master"].discard(client)

# Start the WebSocket server
async def main():
    print("Starting WebSocket server on localhost:8765")
    
    try:
        # Single server handling all connections
        server = await websockets.serve(websocket_handler, "localhost", 8765)
        print("WebSocket server started. Press Ctrl+C to stop.")
        print("Available endpoints:")
        print("  - ws://localhost:8765/local_master (video stream)")
        print("  - ws://localhost:8765/regional_master (notifications)")
        print("  - ws://localhost:8765/acknowledge (acknowledgments)")
        
        # Keep the server running
        await server.wait_closed()
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        
        # Clean up all client connections
        all_clients = set()
        for client_set in clients.values():
            all_clients.update(client_set)
            
        for client in all_clients:
            try:
                await client.close()
            except:
                pass
        
        server.close()
        await server.wait_closed()
        print("Server shut down successfully.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
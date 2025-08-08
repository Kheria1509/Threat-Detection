# Improved threat detection server
import cv2
import json
import asyncio
import websockets
import base64
from ultralytics import YOLO
from collections import deque
from threading import Thread
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import winsound  # Windows-specific audio
    WINDOWS_AUDIO = True
except ImportError:
    WINDOWS_AUDIO = False
    # Alternative for cross-platform audio
    try:
        import pygame
        pygame.mixer.init()
        PYGAME_AUDIO = True
    except ImportError:
        PYGAME_AUDIO = False
        logger.warning("No audio libraries available. Alarm sounds will be disabled.")

# Load models with error handling
def load_models():
    try:
        # Try different possible paths for the models
        model_paths = [
            (r'detect\threat_train\weights\best.pt', r'detect/fire_smoke_train/weights/best.pt'),
            (r'M:/mask/detect/threat_train/weights/best.pt', r'detect/fire_smoke_train/weights/best.pt'),
            ('threat_model.pt', 'fire_smoke_model.pt')  # fallback names
        ]
        
        weapon_model = None
        fire_smoke_model = None
        
        for weapon_path, fire_path in model_paths:
            if os.path.exists(weapon_path) and os.path.exists(fire_path):
                weapon_model = YOLO(weapon_path)
                fire_smoke_model = YOLO(fire_path)
                logger.info(f"Models loaded from: {weapon_path}, {fire_path}")
                break
        
        if weapon_model is None or fire_smoke_model is None:
            logger.error("Could not find model files. Please check model paths.")
            return None, None
            
        return weapon_model, fire_smoke_model
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return None, None

weapon_model, fire_smoke_model = load_models()

# Object classes
weapon_class_names = ["violence", "gun", "knife"]
fire_smoke_class_names = ["fire", "smoke"]

# Store clients and threat status
clients = {
    "local_master": set(),
    "regional_master": set()
}

# Global state variables
threat_detected = False
threat_acknowledged = False
threat_detection_window = deque(maxlen=30)  # Store last 30 frames for threat analysis
alarm_playing = False

# Define thresholds
THREAT_CONFIDENCE_THRESHOLD = 0.50
FIRE_CONFIDENCE_THRESHOLD = 0.50
SMOKE_CONFIDENCE_THRESHOLD = 0.90
ALARM_FILE = 'alarm.mp3'  # Replace with path to your alarm sound file

# Audio handling functions
def play_alarm():
    global alarm_playing
    if alarm_playing:
        return
        
    alarm_playing = True
    logger.info("Playing alarm")
    
    try:
        if WINDOWS_AUDIO and os.path.exists(ALARM_FILE):
            winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME | winsound.SND_ASYNC)
        elif WINDOWS_AUDIO:
            # Fallback to system beep
            winsound.Beep(1000, 1000)  # Frequency, duration in milliseconds
        elif PYGAME_AUDIO and os.path.exists(ALARM_FILE):
            pygame.mixer.music.load(ALARM_FILE)
            pygame.mixer.music.play(-1)  # Loop indefinitely
        else:
            logger.warning("No audio method available for alarm")
    except Exception as e:
        logger.error(f"Error playing alarm: {e}")

def stop_alarm():
    global alarm_playing
    alarm_playing = False
    logger.info("Stopping alarm")
    
    try:
        if WINDOWS_AUDIO:
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif PYGAME_AUDIO:
            pygame.mixer.music.stop()
    except Exception as e:
        logger.error(f"Error stopping alarm: {e}")

# Improved threat detection logic
def is_threat(detections):
    global threat_detection_window, threat_detected
    threat_detection_window.append(detections)
    
    if len(threat_detection_window) < threat_detection_window.maxlen:
        return False  # Not enough data to make a decision
    
    # If a threat is already detected and not acknowledged, continue showing threat
    if threat_detected and not threat_acknowledged:
        return True
    
    # Reset threat status if it was acknowledged
    if threat_acknowledged:
        return False
    
    threat_count = 0
    for frame_detections in threat_detection_window:
        if any(d['class'] in ["gun", "knife", "fire", "smoke"] and d['confidence'] > THREAT_CONFIDENCE_THRESHOLD for d in frame_detections):
            threat_count += 1
    
    # Require threat to be detected in more than half of recent frames
    return threat_count > (threat_detection_window.maxlen // 2)

def detect_objects(frame):
    if weapon_model is None or fire_smoke_model is None:
        logger.warning("Models not loaded, skipping detection")
        return []
        
    results = []

    try:
        # Detect weapons
        weapon_results = weapon_model(frame, stream=True, verbose=False)
        for r in weapon_results:
            if r.boxes is not None:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls < len(weapon_class_names):  # Safety check
                        class_name = weapon_class_names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if confidence > THREAT_CONFIDENCE_THRESHOLD:
                            results.append({
                                "class": class_name,
                                "confidence": confidence,
                                "box": [x1, y1, x2, y2],
                                "color": (0, 0, 255),
                                "timestamp": time.time()
                            })

        # Detect fire and smoke
        fire_results = fire_smoke_model(frame, stream=True, verbose=False)
        for r in fire_results:
            if r.boxes is not None:
                for box in r.boxes:
                    confidence = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls < len(fire_smoke_class_names):  # Safety check
                        class_name = fire_smoke_class_names[cls]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        threshold = FIRE_CONFIDENCE_THRESHOLD if class_name == "fire" else SMOKE_CONFIDENCE_THRESHOLD
                        
                        if confidence > threshold:
                            color = (0, 165, 255) if class_name == "fire" else (128, 128, 128)
                            results.append({
                                "class": class_name,
                                "confidence": confidence,
                                "box": [x1, y1, x2, y2],
                                "color": color,
                                "timestamp": time.time()
                            })

    except Exception as e:
        logger.error(f"Error in object detection: {e}")

    return results

# Improved video source selection
def get_video_source():
    """Try different video sources in order of preference"""
    video_sources = [
        # 'FIGHT_PRACTICE.mp4',
        # 'Explosion.mp4',
        'Gun.mp4',
        # 'Brandon.mp4',
        0  # Webcam as fallback
    ]
    
    for source in video_sources:
        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Using video source: {source}")
                    return cap
                cap.release()
        except Exception as e:
            logger.warning(f"Could not open video source {source}: {e}")
    
    logger.error("No valid video source found")
    return None

# WebSocket server function for video streaming
async def video_stream(websocket, path):
    global threat_detected, threat_acknowledged
    
    user_type = path.strip('/')
    if user_type not in ["local_master", "regional_master"]:
        logger.warning(f"Invalid user type: {user_type}")
        await websocket.close()
        return

    clients[user_type].add(websocket)
    logger.info(f"New {user_type} connected. Total {user_type} connections: {len(clients[user_type])}")
    
    cap = get_video_source()
    if cap is None:
        logger.error("No video source available")
        await websocket.close()
        return
    
    # Set video properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    last_threat_notification = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                # Try to restart video if it's a file
                if isinstance(cap.get(cv2.CAP_PROP_FRAME_COUNT), float):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
                    continue
                else:
                    logger.error("Failed to read from video source")
                    break

            detections = detect_objects(frame)
            previous_threat_status = threat_detected
            threat_detected = is_threat(detections)
            
            # Reset acknowledgment if threat is no longer detected
            if not threat_detected and previous_threat_status:
                threat_acknowledged = False
                stop_alarm()

            # Draw boxes and labels
            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                color = detection["color"]
                label = f"{detection['class']} {detection['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add timestamp to frame
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Encode the frame
            try:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_encoded = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                continue

            # Prepare data to send
            data = {
                "detections": detections,
                "frame": frame_encoded,
                "threat_detected": threat_detected,
                "frame_number": frame_count,
                "timestamp": timestamp
            }

            # Send data to specific client type
            if user_type == "local_master":
                try:
                    await websocket.send(json.dumps(data))
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error sending to local_master: {e}")
                    break

            # Handle threat detection
            if threat_detected and not threat_acknowledged:
                current_time = time.time()
                if current_time - last_threat_notification > 1:  # Avoid spam
                    play_alarm()
                    last_threat_notification = current_time
                    
                    # Start timeout task if not already started
                    asyncio.create_task(threat_timeout())

            # Notify regional masters about threats
            if user_type == "regional_master" or threat_detected:
                notification = {
                    "message": "Threat detected" if threat_detected else "Status update",
                    "threat_detected": threat_detected,
                    "detections": detections,
                    "timestamp": timestamp
                }
                
                # Send to all regional masters
                disconnected_clients = set()
                for regional_master in clients["regional_master"]:
                    try:
                        await regional_master.send(json.dumps(notification))
                    except websockets.exceptions.ConnectionClosed:
                        disconnected_clients.add(regional_master)
                    except Exception as e:
                        logger.error(f"Error sending to regional_master: {e}")
                        disconnected_clients.add(regional_master)
                
                # Remove disconnected clients
                clients["regional_master"] -= disconnected_clients

            frame_count += 1
            await asyncio.sleep(0.033)  # ~30 FPS

    except Exception as e:
        logger.error(f"Error in video_stream: {e}")
    finally:
        cap.release()
        if websocket in clients[user_type]:
            clients[user_type].remove(websocket)
            logger.info(f"{user_type} disconnected. Remaining {user_type} connections: {len(clients[user_type])}")

async def threat_timeout():
    global threat_detected, threat_acknowledged
    
    # Wait for 10 seconds for acknowledgment
    await asyncio.sleep(10)

    if threat_detected and not threat_acknowledged:
        logger.warning("Threat not acknowledged - escalating to regional masters")
        
        # Notify all regional masters
        notification = {
            "message": "Threat not acknowledged - ESCALATED",
            "threat_detected": threat_detected,
            "escalated": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        disconnected_clients = set()
        for regional_master in clients["regional_master"]:
            try:
                await regional_master.send(json.dumps(notification))
            except Exception as e:
                logger.error(f"Error sending escalation to regional_master: {e}")
                disconnected_clients.add(regional_master)
        
        # Remove disconnected clients
        clients["regional_master"] -= disconnected_clients
        
        # Continue alarm
        play_alarm()

async def handle_acknowledgment(websocket, path):
    global threat_acknowledged, threat_detected
    
    logger.info("Acknowledgment handler connected")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get('action') == 'acknowledge_threat':
                    threat_acknowledged = True
                    threat_detected = False  # Reset threat status
                    stop_alarm()
                    logger.info("Threat acknowledged by local master")

                    # Notify all regional masters about threat acknowledgment
                    ack_notification = {
                        "message": "Threat acknowledged by local master",
                        "threat_acknowledged": True,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    disconnected_clients = set()
                    for regional_master in clients["regional_master"]:
                        try:
                            await regional_master.send(json.dumps(ack_notification))
                        except Exception as e:
                            logger.error(f"Error sending ack to regional_master: {e}")
                            disconnected_clients.add(regional_master)
                    
                    # Remove disconnected clients
                    clients["regional_master"] -= disconnected_clients
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received in acknowledgment handler")
            except Exception as e:
                logger.error(f"Error processing acknowledgment: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Acknowledgment connection closed")
    except Exception as e:
        logger.error(f"Error in acknowledgment handler: {e}")

# Improved server startup
async def main():
    logger.info("Starting threat detection server...")
    
    # Check if models are loaded
    if weapon_model is None or fire_smoke_model is None:
        logger.error("Cannot start server without models. Please check model paths.")
        return
    
    try:
        video_server = await websockets.serve(video_stream, "localhost", 8765)
        ack_server = await websockets.serve(handle_acknowledgment, "localhost", 8766)
        
        logger.info("Servers started successfully:")
        logger.info("- Video/Detection server: ws://localhost:8765")
        logger.info("- Acknowledgment server: ws://localhost:8766")
        logger.info("\nAvailable endpoints:")
        logger.info("- ws://localhost:8765/local_master (video stream)")
        logger.info("- ws://localhost:8765/regional_master (notifications)")
        logger.info("- ws://localhost:8766 (acknowledgments)")
        
        # Keep the server running
        await asyncio.gather(
            video_server.wait_closed(),
            ack_server.wait_closed()
        )
        
    except Exception as e:
        logger.error(f"Failed to start servers: {e}")
    finally:
        # Cleanup
        stop_alarm()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer shutdown by user")
        stop_alarm()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        cv2.destroyAllWindows()
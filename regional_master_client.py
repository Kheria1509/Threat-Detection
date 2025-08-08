import asyncio
import websockets
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def connect_to_server():
    uri = "ws://localhost:8765/regional_master"
    retry_delay = 5  # seconds between retry attempts
    
    while True:
        try:
            async with websockets.connect(uri) as websocket:
                logger.info("Connected to threat detection server")
                await handle_messages(websocket)
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection lost. Retrying in %d seconds...", retry_delay)
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logger.error("Error: %s. Retrying in %d seconds...", str(e), retry_delay)
            await asyncio.sleep(retry_delay)

async def handle_messages(websocket):
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            # Handle different types of messages
            if "threat_detected" in data:
                if data["threat_detected"]:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logger.warning("⚠️ [%s] Threat detected!", timestamp)
                    
                    # If there are detections, show them
                    if "detections" in data:
                        for detection in data["detections"]:
                            logger.info("   - %s (confidence: %.2f)", 
                                      detection["class"], 
                                      detection["confidence"])
            
            if "message" in data:
                if data["message"] == "Threat not acknowledged":
                    logger.critical("🚨 ALERT: Threat not acknowledged by local master!")
                    # You could add additional actions here, like sending SMS or other notifications
                
                elif data["message"] == "Threat acknowledged":
                    logger.info("✓ Threat has been acknowledged by local master")
                
                else:
                    logger.info("Message from server: %s", data["message"])

    except websockets.exceptions.ConnectionClosed:
        logger.error("Connection to server closed")
    except Exception as e:
        logger.error("Error handling messages: %s", str(e))

async def main():
    try:
        logger.info("Starting Regional Master Client...")
        logger.info("Connecting to threat detection server...")
        await connect_to_server()
    except KeyboardInterrupt:
        logger.info("Client shutdown by user")
    except Exception as e:
        logger.error("Fatal error: %s", str(e))

if __name__ == "__main__":
    asyncio.run(main())

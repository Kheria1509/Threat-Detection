import asyncio
import websockets
import json

async def receive_notifications():
    uri = "ws://localhost:8765/regional_master"

    async with websockets.connect(uri) as websocket:
        print("Connected to the server as Regional Master.")
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)

                if data.get('threat_detected'):
                    print("Threat detected!")
                    print("Waiting for local master acknowledgment...")

                if data.get('message') == "Threat not acknowledged":
                    print("Threat was not acknowledged by local master.")
                    await acknowledge_threat(websocket)

                if data.get('message') == "Threat acknowledged":
                    print("Threat has been acknowledged by local master.")
        
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            return

async def acknowledge_threat(websocket):
    acknowledgment = json.dumps({"action": "acknowledge_threat"})
    await websocket.send(acknowledgment)
    print("Acknowledged the threat as Regional Master.")

if __name__ == "__main__":
    asyncio.run(receive_notifications())

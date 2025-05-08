import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, Optional

import redis.asyncio as redis # Using asyncio version of redis client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AuraGUIMonitor")

# --- Configuration ---
@dataclass
class Config:
    event_broker_url: str = "redis://localhost:6379/0"
    new_request_event_topic: str = "aura.requests.new"
    gui_host: str = "127.0.0.1"
    gui_port: int = 5001

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            field_names = {f.name for f in fields(cls)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
            instance = cls(**filtered_dict) # Use kwargs to init
            logger.info(f"Configuration successfully loaded from {config_path}")
            return instance
        except FileNotFoundError:
            logger.warning(f"Config file '{config_path}' not found. Using default settings.")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{config_path}': {e}. Using default settings.")
            return cls()
        except Exception as e:
            logger.error(f"Unexpected error loading config '{config_path}': {e}. Using default settings.")
            return cls()

    def to_file(self, config_path: str) -> None:
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to '{config_path}': {e}")

# --- L-EBM Client (Subscriber) ---
class EventSubscriber:
    def __init__(self, config: Config, websocket_manager):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.websocket_manager = websocket_manager
        self._stop_event = asyncio.Event()

    async def connect(self):
        try:
            self.redis_client = redis.Redis.from_url(self.config.event_broker_url, decode_responses=True)
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            logger.info(f"Successfully connected to L-EBM at {self.config.event_broker_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to L-EBM: {e}")
            self.redis_client = None
            return False

    async def listen(self):
        if not self.redis_client or not self.pubsub:
            logger.error("Not connected to L-EBM. Cannot listen.")
            return

        await self.pubsub.subscribe(self.config.new_request_event_topic)
        logger.info(f"Subscribed to L-EBM topic: {self.config.new_request_event_topic}")

        while not self._stop_event.is_set():
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0) # Timeout to allow stop_event check
                if message and message["type"] == "message":
                    logger.info(f"Received message from L-EBM: {message['data']}")
                    try:
                        event_data = json.loads(message['data'])
                        await self.websocket_manager.broadcast(event_data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from L-EBM message: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error processing message or broadcasting: {e}")
                await asyncio.sleep(0.01) # Short sleep to prevent tight loop if no messages
            except redis.exceptions.ConnectionError as e:
                logger.error(f"L-EBM connection error while listening: {e}. Attempting to reconnect...")
                await self.disconnect() # Clean up old connection
                await asyncio.sleep(5) # Wait before reconnecting
                if not await self.connect():
                    logger.error("Reconnect failed. Stopping listener.")
                    break
                else:
                    await self.pubsub.subscribe(self.config.new_request_event_topic) # Re-subscribe
                    logger.info("Reconnected to L-EBM and re-subscribed.")
            except Exception as e:
                logger.error(f"Unexpected error in L-EBM listener: {e}")
                # Depending on error, might need more specific handling or break
                await asyncio.sleep(1)
        logger.info("L-EBM listener stopped.")

    async def disconnect(self):
        self._stop_event.set()
        if self.pubsub:
            try:
                await self.pubsub.unsubscribe(self.config.new_request_event_topic)
                await self.pubsub.close() # Close pubsub before client
                logger.info("Unsubscribed and closed L-EBM pubsub.")
            except Exception as e:
                logger.error(f"Error closing L-EBM pubsub: {e}")
            self.pubsub = None
        if self.redis_client:
            try:
                await self.redis_client.close()
                await self.redis_client.connection_pool.disconnect() # Ensure pool is disconnected
                logger.info("Closed L-EBM client connection.")
            except Exception as e:
                logger.error(f"Error closing L-EBM client: {e}")
            self.redis_client = None

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected: {websocket.client}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected: {websocket.client}")

    async def broadcast(self, data: Dict[str, Any]):
        message_json = json.dumps(data)
        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except WebSocketDisconnect:
                disconnected_clients.append(connection)
                logger.warning(f"WebSocket client {connection.client} disconnected during broadcast.")
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket client {connection.client}: {e}")
                disconnected_clients.append(connection) # Assume problematic
        
        for client in disconnected_clients:
            if client in self.active_connections: # Check if not already removed by another task
                 self.active_connections.remove(client)

# --- FastAPI Application Setup ---
app = FastAPI()
manager = ConnectionManager()

# Ensure static and templates directories exist
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, "static"), exist_ok=True)
os.makedirs(os.path.join(script_dir, "templates"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(script_dir, "static")), name="static")

# Global variable for config and subscriber
config: Optional[Config] = None
subscriber: Optional[EventSubscriber] = None
listener_task: Optional[asyncio.Task] = None

@app.on_event("startup")
async def startup_event():
    global config, subscriber, listener_task
    config_path = os.getenv("AURA_GUI_CONFIG_PATH", "config_gui.json")
    config = Config.from_file(config_path)
    if not os.path.exists(config_path):
        config.to_file(config_path) # Save default if not found

    subscriber = EventSubscriber(config, manager)
    if await subscriber.connect():
        listener_task = asyncio.create_task(subscriber.listen())
        logger.info("L-EBM listener task created and started.")
    else:
        logger.error("Failed to connect to L-EBM on startup. Listener not started.")

@app.on_event("shutdown")
async def shutdown_event():
    global listener_task, subscriber
    logger.info("GUI Monitor shutting down...")
    if listener_task and not listener_task.done():
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            logger.info("L-EBM listener task cancelled.")
    if subscriber:
        await subscriber.disconnect()
    logger.info("GUI Monitor shutdown complete.")

@app.get("/", response_class=HTMLResponse)
async def get_gui():
    # HTML content will be created in a separate step/file
    html_content_path = os.path.join(script_dir, "templates", "index.html")
    if os.path.exists(html_content_path):
        with open(html_content_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        # Fallback minimal HTML if file not found
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
            <head>
                <title>AURA Event Monitor</title>
            </head>
            <body>
                <h1>AURA Event Monitor</h1>
                <p>Waiting for events... (HTML file not found, using fallback)</p>
                <ul id="event-list"></ul>
                <script>
                    const eventList = document.getElementById('event-list');
                    const ws = new WebSocket(`ws://${location.host}/ws`);
                    ws.onmessage = function(event) {
                        const listItem = document.createElement('li');
                        listItem.textContent = event.data;
                        eventList.appendChild(listItem);
                        window.scrollTo(0, document.body.scrollHeight);
                    };
                    ws.onopen = () => console.log("WebSocket connected");
                    ws.onclose = () => console.log("WebSocket disconnected");
                    ws.onerror = (error) => console.error("WebSocket error:", error);
                </script>
            </body>
        </html>
        """)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive, actual data is pushed by L-EBM listener via manager.broadcast
            await websocket.receive_text() # This can be used for client-to-server messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error for client {websocket.client}: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Load config for uvicorn host/port settings directly for main execution
    # This is a bit redundant with startup_event but ensures uvicorn uses correct values if run directly.
    initial_config_path = os.getenv("AURA_GUI_CONFIG_PATH", "config_gui.json")
    if not os.path.exists(initial_config_path):
        Config().to_file(initial_config_path) # Create default if it doesn't exist
    
    loaded_config = Config.from_file(initial_config_path)
    
    logger.info(f"Starting AURA GUI Monitor on {loaded_config.gui_host}:{loaded_config.gui_port}")
    uvicorn.run(app, host=loaded_config.gui_host, port=loaded_config.gui_port, log_level="info")

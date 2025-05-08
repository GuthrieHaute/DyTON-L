#!/usr/bin/env python3
# AURA-DyTON Core Processing Node
# This component consumes NewRequestEvents from the L-EBM, processes them through Ollama, and logs responses.

import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, List, Optional, Any, Deque
from collections import deque
import requests
import argparse

# --- Logging Setup ---
def setup_logging(log_file, console_level_str, file_level_str):
    """Configure logging with file and console handlers."""
    log_file_handler = logging.FileHandler(log_file)
    log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Custom chat level for user interactions
    CHAT_LEVEL_NUM = 25
    logging.addLevelName(CHAT_LEVEL_NUM, "CHAT")
    def chat_log(self, message, *args, **kws):
        if self.isEnabledFor(CHAT_LEVEL_NUM):
            self._log(CHAT_LEVEL_NUM, message, args, **kws)
    logging.Logger.chat = chat_log

    # Configure console handler with special formatting for CHAT level
    log_stream_handler = logging.StreamHandler()
    class ChatLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno == CHAT_LEVEL_NUM or record.levelno >= logging.ERROR
    log_stream_handler.addFilter(ChatLevelFilter())
    log_stream_handler.setFormatter(logging.Formatter('%(message)s'))

    # Set levels based on input parameters
    console_level = CHAT_LEVEL_NUM if console_level_str == "CHAT" else getattr(logging, console_level_str, logging.INFO)
    log_stream_handler.setLevel(console_level)
    file_level = getattr(logging, file_level_str, logging.INFO)
    log_file_handler.setLevel(file_level)

    # Configure root logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(log_file_handler)
    root_logger.addHandler(log_stream_handler)
    min_level = min(file_level, console_level if console_level != CHAT_LEVEL_NUM else logging.INFO)
    if console_level == CHAT_LEVEL_NUM:
        min_level = min(min_level, CHAT_LEVEL_NUM)
    root_logger.setLevel(min_level)

    # Create and configure specific loggers
    logger = logging.getLogger("aura_core_processor")
    chat_logger = logging.getLogger("chat_interface")
    chat_logger.propagate = True
    chat_logger.setLevel(CHAT_LEVEL_NUM)

    return logger, chat_logger

# --- Configuration ---
@dataclass
class Config:
    """Configuration for the AURA Core Processing Node."""
    # L-EBM Connection settings
    event_broker_url: str = "redis://localhost:6379/0"
    new_request_event_topic: str = "aura.requests.new"
    speak_request_event_topic: str = "aura.tts.speak_request"
    
    # Ollama Settings
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "mrpickles2:latest"
    chat_history_size: int = 15
    ollama_num_ctx: int = 0
    ollama_timeout: int = 90
    
    # Logging Settings
    log_file: str = "core_processor.log"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    # LLM Interaction
    system_message: str = "You are AURA, an advanced AI assistant. Be helpful, concise, and friendly."

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file, with error handling."""
        logger = logging.getLogger("aura_core_processor")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            field_names = {f.name for f in fields(cls)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
            instance = cls(**filtered_dict)  # Use kwargs to initialize
            logger.info(f"Configuration successfully loaded from {config_path}")
            return instance
        except FileNotFoundError:
            logger.warning(f"Configuration file '{config_path}' not found. Using default settings.")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{config_path}': {e}. Using default settings.")
            return cls()
        except Exception as e:
            logger.error(f"Unexpected error loading config '{config_path}': {e}. Using default settings.")
            return cls()

    def to_file(self, config_path: str) -> None:
        """Save the current configuration to a JSON file."""
        logger = logging.getLogger("aura_core_processor")
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, indent=4)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to '{config_path}': {e}")


# --- OllamaChat Class ---
class OllamaChat:
    """Handles interaction with the Ollama LLM API."""
    
    def __init__(self, config: Config):
        """Initialize OllamaChat with configuration."""
        self.logger = logging.getLogger("aura_core_processor")
        self.chat_logger = logging.getLogger("chat_interface")
        self.config = config
        self.chat_history: Deque[Dict[str, str]] = deque(maxlen=config.chat_history_size)
        self.is_available = False
    
    def check_availability(self) -> bool:
        """Check if the Ollama server is reachable and operational."""
        self.logger.info(f"Checking Ollama availability at {self.config.ollama_url}...")
        
        if not self.config.ollama_url:
            self.logger.warning("Ollama URL not configured.")
            self.is_available = False
            return False
        
        try:
            # Extract base URL from API endpoint
            base_url_parts = self.config.ollama_url.split('/api/')
            health_check_url = base_url_parts[0]
            if not health_check_url.startswith(('http://', 'https://')):
                self.logger.error(f"Invalid Ollama URL format: {self.config.ollama_url}")
                self.is_available = False
                return False
            
            # Try HEAD request first (more efficient)
            try:
                response = requests.head(health_check_url, timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException:
                self.logger.debug("HEAD failed, trying GET.")
                response = requests.get(health_check_url, timeout=5)
                response.raise_for_status()
            
            self.logger.info(f"Ollama server reachable at {health_check_url}.")
            self.is_available = True
            
            # Additional check to verify model availability
            self.logger.info(f"Verifying Ollama model: {self.config.ollama_model}")
            # This is just a simple start of a query to verify the model exists
            test_payload = {
                "model": self.config.ollama_model,
                "messages": [{"role": "system", "content": "Test"}],
                "stream": False
            }
            response = requests.post(
                self.config.ollama_url,
                json=test_payload,
                timeout=10
            )
            if response.status_code == 404:
                self.logger.error(f"Model '{self.config.ollama_model}' not found on Ollama server.")
                self.is_available = False
                return False
            elif not response.ok:
                self.logger.warning(f"Model check received non-OK response: {response.status_code}")
                # We'll still return True since the server is up, but log the concern
            
            return True
        except requests.exceptions.Timeout:
            self.logger.error(f"Ollama check timed out at {health_check_url}.")
            self.is_available = False
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error(f"Ollama connection refused at {health_check_url}. Is it running?")
            self.is_available = False
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama check failed: {e}")
            self.is_available = False
            return False
        except Exception as e:
            self.logger.error(f"Ollama availability check unexpected error: {e}")
            self.is_available = False
            return False
    
    def add_history(self, role: str, content: str):
        """Add a message to the chat history deque."""
        if role not in ["user", "assistant", "system"]:
            self.logger.warning(f"Invalid role for chat history: {role}")
            return
        self.chat_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear the chat history."""
        self.logger.info("Clearing Ollama chat history.")
        self.chat_history.clear()
    
    def query(self, prompt: str, role: str = "user", add_prompt_to_history: bool = True, 
              add_response_to_history: bool = True, system_message: Optional[str] = None) -> Optional[str]:
        """
        Send a prompt to Ollama and return the response.
        
        Args:
            prompt: The text prompt to send to Ollama
            role: The role for the prompt message (usually "user")
            add_prompt_to_history: Whether to add the prompt to chat history
            add_response_to_history: Whether to add Ollama's response to chat history
            system_message: Optional system message to override the default
            
        Returns:
            The response text from Ollama, or None if the query failed
        """
        if not self.is_available:
            self.logger.warning("Ollama query skipped: Server not available.")
            return None
        
        # Use current history and add system message if provided
        current_messages = list(self.chat_history)
        system_msg = system_message or self.config.system_message
        
        if current_messages and current_messages[0]['role'] == 'system':
            current_messages[0] = {"role": "system", "content": system_msg}
        else:
            current_messages.insert(0, {"role": "system", "content": system_msg})
        
        # Add the current prompt
        current_messages.append({"role": role, "content": prompt})
        
        # Add to history if specified (will be removed later if response fails and add_response_to_history=False)
        if add_prompt_to_history:
            self.add_history(role, prompt)
        
        # Build the API payload
        payload = {
            "model": self.config.ollama_model,
            "messages": current_messages,
            "stream": False,
            "options": {}
        }
        if self.config.ollama_num_ctx > 0:
            payload["options"]["num_ctx"] = self.config.ollama_num_ctx
            self.logger.debug(f"Requesting Ollama num_ctx: {self.config.ollama_num_ctx}")
        
        # Log the query information
        self.logger.info(
            f"Querying Ollama model '{self.config.ollama_model}' "
            f"(History: {len(self.chat_history)}, Prompt: {len(prompt)} chars)..."
        )
        self.chat_logger.chat(f"USER: {prompt}")
        
        # Send the request to Ollama
        query_start_time = time.time()
        try:
            response = requests.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.config.ollama_timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract response content from Ollama's response format
            response_content = None
            if 'message' in data and isinstance(data['message'], dict) and 'content' in data['message']:
                response_content = data['message']['content']
            elif 'response' in data:
                response_content = data['response']
            elif 'error' in data:
                self.logger.error(f"Ollama API error: {data['error']}")
                return None
            
            if response_content:
                response_content = response_content.strip()
                query_latency = time.time() - query_start_time
                self.logger.info(
                    f"Ollama query successful ({query_latency:.2f}s). "
                    f"Response length: {len(response_content)} chars"
                )
                self.chat_logger.chat(f"AURA: {response_content}")
                
                if add_response_to_history:
                    self.add_history("assistant", response_content)
                return response_content
            else:
                self.logger.warning(f"Ollama empty/unexpected response format: {data}")
                return None
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Ollama query timed out ({self.config.ollama_timeout}s).")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama query failed: {e}")
            return None
        except Exception as e:
            self.logger.exception("Ollama query unexpected error")
            return None
        finally:
            # If we're not supposed to keep response in history but added the prompt,
            # and either failed to get a response or won't be adding it to history,
            # remove the prompt from history
            if add_prompt_to_history and not add_response_to_history and self.chat_history and \
               self.chat_history[-1]['role'] == role and self.chat_history[-1]['content'] == prompt:
                try:
                    self.chat_history.pop()
                except IndexError:
                    pass


# --- L-EBM Client (Subscriber) ---
class EventSubscriber:
    """Subscribe to L-EBM topics and process events."""
    
    def __init__(self, config: Config, ollama_chat: OllamaChat):
        """
        Initialize the event subscriber.
        
        Args:
            config: Configuration object
            ollama_chat: OllamaChat instance for processing requests
        """
        self.logger = logging.getLogger("aura_core_processor")
        self.config = config
        self.ollama_chat = ollama_chat
        self.redis_client = None
        self.pubsub = None
        self.running = False
        self._stop_event = asyncio.Event()
    
    async def connect(self) -> bool:
        """Connect to the L-EBM (e.g., Redis)."""
        self.logger.info(f"Connecting to L-EBM at {self.config.event_broker_url}...")
        
        try:
            # Dynamic import to avoid requiring redis for development/testing
            import redis.asyncio as redis
            self.redis_client = redis.Redis.from_url(
                self.config.event_broker_url, 
                decode_responses=True
            )
            await self.redis_client.ping()
            self.pubsub = self.redis_client.pubsub()
            self.logger.info(f"Successfully connected to L-EBM at {self.config.event_broker_url}")
            return True
        except ImportError:
            self.logger.error("Redis library not installed. Run 'pip install redis'")
            return False
        except Exception as e:
            self.logger.error(f"Failed to connect to L-EBM: {e}")
            self.redis_client = None
            return False
    
    async def publish(self, topic: str, event_data: Dict[str, Any]) -> bool:
        """
        Publish an event to the specified L-EBM topic.
        
        Args:
            topic: The topic to publish to
            event_data: The event data to publish (will be serialized to JSON)
            
        Returns:
            True if publication was successful, False otherwise
        """
        if not self.redis_client:
            self.logger.error("Not connected to L-EBM. Cannot publish.")
            return False
            
        try:
            event_json = json.dumps(event_data)
            result = await self.redis_client.publish(topic, event_json)
            self.logger.debug(f"Published event to {topic}, delivery count: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish event to {topic}: {e}")
            return False
    
    async def listen(self):
        """Listen for events on the configured topic and process them."""
        if not self.redis_client or not self.pubsub:
            self.logger.error("Not connected to L-EBM. Cannot listen.")
            return
        
        # Subscribe to the configured topic
        await self.pubsub.subscribe(self.config.new_request_event_topic)
        self.logger.info(f"Subscribed to L-EBM topic: {self.config.new_request_event_topic}")
        
        self.running = True
        
        # Main event loop
        while not self._stop_event.is_set():
            try:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0  # Short timeout to check for stop_event regularly
                )
                
                if message and message["type"] == "message":
                    self.logger.debug(f"Received message from L-EBM: {message['data']}")
                    await self.process_event(message['data'])
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in L-EBM listener: {e}")
                # Brief pause before retry
                await asyncio.sleep(1)
        
        self.running = False
        self.logger.info("L-EBM listener stopped.")
    
    async def process_event(self, event_data_str: str):
        """
        Process a received event from the L-EBM.
        
        Args:
            event_data_str: The event payload as a JSON string
        """
        try:
            # Parse the event data
            event_data = json.loads(event_data_str)
            
            # Validate that it's a NewRequestEvent
            if "input_type" not in event_data or "transcribed_text" not in event_data:
                self.logger.warning(f"Received malformed event, missing required fields: {event_data}")
                return
            
            # Log the received request
            req_id = event_data.get("request_id", "unknown")
            timestamp = event_data.get("timestamp", datetime.datetime.now(
                datetime.timezone.utc).isoformat())
            user_id = event_data.get("user_id", "default_user")
            transcribed_text = event_data["transcribed_text"]
            
            self.logger.info(
                f"Processing request: ID={req_id}, User={user_id}, "
                f"Timestamp={timestamp}, Text='{transcribed_text}'"
            )
            
            # Query Ollama with the transcribed text
            response = self.ollama_chat.query(transcribed_text)
            
            if response:
                self.logger.info(f"Successfully processed request {req_id}")
                
                # Generate a unique event ID for the SpeakTextEvent
                event_id = f"speak_{int(time.time()*1000)}_{hash(response) % 10000:04d}"
                current_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                # Create the SpeakTextEvent payload
                speak_event = {
                    "event_id": event_id,
                    "timestamp": current_timestamp,
                    "text_to_speak": response,
                    "correlation_id": req_id,
                    "user_id": user_id,
                    # Optional fields that could be added in future enhancements:
                    # "voice_id": "default",
                    # "speaking_rate": 1.0,
                    # "priority": 1
                }
                
                # Publish the SpeakTextEvent
                success = await self.publish(
                    self.config.speak_request_event_topic,
                    speak_event
                )
                
                if success:
                    self.logger.info(
                        f"Published SpeakTextEvent for request {req_id} "
                        f"(Event ID: {event_id}, Text length: {len(response)} chars)"
                    )
                else:
                    self.logger.error(f"Failed to publish SpeakTextEvent for request {req_id}")
            else:
                self.logger.error(f"Failed to get response from Ollama for request {req_id}")
            
        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse JSON from event: {event_data_str}")
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
    
    async def disconnect(self):
        """Disconnect from the L-EBM."""
        self.logger.info("Disconnecting from L-EBM...")
        self._stop_event.set()
        
        # Wait briefly for the listen loop to notice the stop event
        await asyncio.sleep(0.1)
        
        if self.pubsub:
            try:
                await self.pubsub.unsubscribe(self.config.new_request_event_topic)
                await self.pubsub.close()
                self.logger.info("Unsubscribed and closed L-EBM pubsub.")
            except Exception as e:
                self.logger.error(f"Error closing L-EBM pubsub: {e}")
            self.pubsub = None
        
        if self.redis_client:
            try:
                await self.redis_client.close()
                self.logger.info("Closed Redis client connection.")
            except Exception as e:
                self.logger.error(f"Error closing Redis client: {e}")
            self.redis_client = None


# --- Main Application ---
class CoreProcessorApp:
    """Main application class for the AURA Core Processing Node."""
    
    def __init__(self, config_path: str = "config_core.json"):
        """Initialize the application with the specified config file."""
        # Load configuration
        self.config_path = config_path
        self.config = Config()
        if os.path.exists(config_path):
            self.config = Config.from_file(config_path)
        else:
            self.config.to_file(config_path)  # Create default config file
        
        # Set up logging
        self.logger, self.chat_logger = setup_logging(
            self.config.log_file,
            self.config.console_log_level,
            self.config.file_log_level
        )
        
        # Components
        self.ollama_chat = OllamaChat(self.config)
        self.event_subscriber = None  # Will be initialized in run()
        
        # State
        self.listener_task = None
        self.running = False
    
    async def run(self):
        """Run the core processor application."""
        self.logger.info("Starting AURA Core Processing Node...")
        
        # Check Ollama availability
        if not self.ollama_chat.check_availability():
            self.logger.error(
                "Ollama is not available. Please ensure Ollama is running "
                f"at {self.config.ollama_url} with the model {self.config.ollama_model} loaded."
            )
            return
        
        # Create and connect the event subscriber
        self.event_subscriber = EventSubscriber(self.config, self.ollama_chat)
        if not await self.event_subscriber.connect():
            self.logger.error(
                f"Failed to connect to L-EBM at {self.config.event_broker_url}. "
                "Please check the connection and try again."
            )
            return
        
        # Start listening for events
        self.running = True
        self.listener_task = asyncio.create_task(self.event_subscriber.listen())
        self.logger.info(
            f"Core Processing Node started. Listening for events on topic: "
            f"{self.config.new_request_event_topic}"
        )
        
        try:
            # Run until interrupted
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.info("Application task cancelled.")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Perform a clean shutdown of the application."""
        self.logger.info("Shutting down Core Processing Node...")
        self.running = False
        
        if self.event_subscriber:
            await self.event_subscriber.disconnect()
        
        if self.listener_task and not self.listener_task.done():
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Core Processing Node shutdown complete.")


async def main_async():
    """Async entry point for the application."""
    parser = argparse.ArgumentParser(
        description="AURA Core Processing Node",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_core.json",
        help="Path to the configuration file"
    )
    args = parser.parse_args()
    
    app = CoreProcessorApp(args.config)
    
    # Set up signal handling for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(app.shutdown()))
    
    await app.run()


def main():
    """Main entry point for the application."""
    if sys.platform == 'win32':
        # Windows-specific event loop policy
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt. Exiting...")
    except Exception as e:
        print(f"Error in main: {e}")


if __name__ == "__main__":
    main()

import os
import sys
import time
import json
import torch
import sounddevice as sd
import logging
import argparse
import threading
import datetime
import uuid
import queue
from dataclasses import dataclass, fields, asdict
from typing import Optional, Dict, Any, List, Union

# --- Setup Logging ---
log_file_handler = logging.FileHandler("tts_output_node.log")
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

log_stream_handler = logging.StreamHandler()
log_stream_handler.setFormatter(logging.Formatter('%(message)s'))

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_stream_handler)
root_logger.setLevel(logging.INFO)

logger = logging.getLogger("tts_output_node")


# --- Configuration Class ---
@dataclass
class Config:
    """Configuration settings for the TTS Output Node."""
    # L-EBM settings
    event_broker_url: str = "redis://localhost:6379/0"
    speak_request_event_topic: str = "aura.tts.speak_request"
    tts_status_event_topic: str = "aura.tts.status"
    
    # TTS Settings
    tts_voice_id: str = "em_alex.pt"
    voices_directory: str = "voices"
    tts_sample_rate: int = 24000  # Default sample rate for TTS output
    tts_speed: float = 1.0        # Speed multiplier for TTS
    
    # Audio output settings
    output_device_index: Optional[int] = None

    def load_from_file(self, file_path: str) -> bool:
        """Loads configuration from a JSON file."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Config file '{file_path}' not found. Using defaults and saving.")
                self.save_to_file(file_path)
                return False
            
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            # Update dataclass fields from the loaded dictionary
            for field in fields(self):
                if field.name in config_dict:
                    setattr(self, field.name, config_dict[field.name])
                    
            logger.info(f"Configuration successfully loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
            
    def save_to_file(self, file_path: str) -> bool:
        """Saves current configuration to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False


# --- Event Broker Client ---
class EventBrokerClient:
    """Handles connection to the L-EBM and subscribes/publishes to topics."""
    def __init__(self, config: Config):
        self.config = config
        self.connected = False
        self.subscriber_thread = None
        self.running = False
        self.callback = None
        
    def connect(self) -> bool:
        """Connect to the event broker middleware."""
        # For Phase 1, we'll simulate the connection to L-EBM
        try:
            logger.info(f"Attempting to connect to L-EBM at {self.config.event_broker_url} (Attempt 1/3)...")
            # Simulate connection success
            self.connected = True
            logger.info(f"L-EBM connection simulated for {self.config.event_broker_url} " +
                      f"(Topic: {self.config.speak_request_event_topic}).")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to L-EBM: {e}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from the event broker middleware."""
        if self.connected:
            # Stop the subscriber thread first
            self.running = False
            if self.subscriber_thread and self.subscriber_thread.is_alive():
                self.subscriber_thread.join(timeout=5.0)
                
            # Simulate disconnection
            logger.info("Disconnecting from L-EBM...")
            self.connected = False
            logger.info("Disconnected from L-EBM (simulated).")
    
    def subscribe(self, callback) -> bool:
        """Subscribe to the speak_request_event_topic and set up a callback for received events."""
        if not self.connected:
            logger.error("Cannot subscribe: Not connected to L-EBM")
            return False
            
        self.callback = callback
        self.running = True
        
        # Start a separate thread to simulate receiving events
        self.subscriber_thread = threading.Thread(
            target=self._subscription_handler,
            daemon=True
        )
        self.subscriber_thread.start()
        
        logger.info(f"Subscribed to topic: {self.config.speak_request_event_topic}")
        return True
        
    def _subscription_handler(self) -> None:
        """Background thread that simulates receiving events from L-EBM."""
        logger.info(f"Subscription handler started for topic: {self.config.speak_request_event_topic}")
        while self.running:
            # In a real implementation, this would wait for messages from the broker
            # For simulation, we just sleep and continue the loop
            time.sleep(0.1)
            
        logger.info("Subscription handler terminated")
    
    def publish(self, topic: str, payload: Dict[str, Any]) -> bool:
        """Publish an event to the specified topic."""
        if not self.connected:
            logger.error("Cannot publish: Not connected to L-EBM")
            return False
            
        try:
            # Simulate publishing
            logger.info(f"L-EBM SIMULATED PUBLISH to topic '{topic}' (Attempt 1): {json.dumps(payload)}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to L-EBM: {e}")
            return False


# --- TTS Engine ---
class TTSEngine:
    """Handles text-to-speech synthesis."""
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.voice_samples = {}
        self.tts_queue = queue.Queue()
        self.speaking = False
        self.tts_thread = None
        self.running = False
        self.voice_path = os.path.join(self.config.voices_directory, self.config.tts_voice_id)
    
    def initialize(self) -> bool:
        """Initialize the TTS engine."""
        logger.info("--- Initializing TTS Engine ---")
        try:
            # Check for CUDA availability
            cuda_ok = torch.cuda.is_available()
            logger.info(f"CUDA Available: {cuda_ok}" + 
                      (f" (Device: {torch.cuda.get_device_name(0)})" if cuda_ok else ""))
            
            # Import TTS libraries dynamically to handle cases where they might not be installed
            try:
                import torch
                from tortoise.api import TextToSpeech
                from tortoise.utils.audio import load_audio, load_voice
                
                # Verify the voice file exists
                if not os.path.exists(self.voice_path):
                    logger.error(f"Voice file not found: {self.voice_path}")
                    return False
                
                # Load TTS model
                logger.info("Loading TTS model...")
                self.tts = TextToSpeech(device=self.device)
                logger.info("TTS model loaded successfully.")
                
                # Load voice samples
                logger.info(f"Loading voice from: {self.voice_path}")
                self.voice_samples = load_voice(self.voice_path)
                logger.info("Voice samples loaded successfully.")
                
                # Start TTS thread
                self.running = True
                self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
                self.tts_thread.start()
                logger.info("TTS worker thread started.")
                
                logger.info("--- TTS Engine Initialization Complete ---")
                return True
                
            except ImportError as ie:
                logger.error(f"Failed to import TTS libraries: {ie}")
                logger.error("Please install the required packages: pip install torchaudio transformers tortoise-tts")
                return False
                
        except Exception as e:
            logger.exception(f"Error initializing TTS Engine: {e}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown the TTS engine."""
        logger.info("Shutting down TTS Engine...")
        self.running = False
        if self.tts_thread and self.tts_thread.is_alive():
            self.tts_thread.join(timeout=5.0)
        logger.info("TTS Engine shutdown complete.")
    
    def speak(self, text: str, speaking_rate: float = None, priority: int = 0) -> str:
        """Queue text to be spoken."""
        if not text or not text.strip():
            logger.warning("Empty text received for TTS, ignoring.")
            return ""
        
        # Generate a unique ID for this speech request
        speech_id = str(uuid.uuid4())
        
        # Use default speaking rate if none provided
        if speaking_rate is None:
            speaking_rate = self.config.tts_speed
        
        # Queue the speech request
        self.tts_queue.put({
            'id': speech_id,
            'text': text.strip(),
            'speaking_rate': speaking_rate,
            'priority': priority,
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
        })
        
        logger.debug(f"TTS: Text queued for speaking (ID: {speech_id}, Priority: {priority})")
        return speech_id
    
    def _tts_worker(self) -> None:
        """Worker thread that processes TTS requests from the queue."""
        logger.info("TTS worker thread started")
        while self.running:
            try:
                # Get the next speech request from the queue
                if self.tts_queue.empty():
                    time.sleep(0.1)
                    continue
                
                speech_req = self.tts_queue.get()
                speech_id = speech_req['id']
                text = speech_req['text']
                speaking_rate = speech_req['speaking_rate']
                
                logger.info(f"TTS: Processing speech request: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
                # Mark as speaking
                self.speaking = True
                
                # Generate speech audio using Tortoise TTS
                try:
                    logger.debug("Generating TTS audio...")
                    start_time = time.time()
                    
                    # Generate speech
                    gen_audio = self.tts.tts(
                        text,
                        voice_samples=self.voice_samples,
                        k=1,
                        # Additional parameters can be added here
                    )
                    
                    # Apply speaking rate (time stretching)
                    if speaking_rate != 1.0:
                        import librosa
                        logger.debug(f"Applying speaking rate: {speaking_rate}")
                        gen_audio = librosa.effects.time_stretch(gen_audio, rate=speaking_rate)
                    
                    end_time = time.time()
                    logger.info(f"TTS generation complete in {end_time - start_time:.2f}s")
                    
                    # Play the audio
                    logger.debug("Playing TTS audio...")
                    sd.play(gen_audio, self.config.tts_sample_rate, device=self.config.output_device_index)
                    sd.wait()  # Wait until audio playback is done
                    
                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                
                # Mark as no longer speaking
                self.speaking = False
                
                # Mark the queue item as done
                self.tts_queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.exception(f"Error in TTS worker: {e}")
                time.sleep(0.5)  # Avoid tight loop in case of persistent errors
        
        logger.info("TTS worker thread terminated")


# --- TTS Output Node ---
class TTSOutputNode:
    """Main orchestrator for the TTS Output Node."""
    def __init__(self, config: Config):
        self.config = config
        self.event_broker_client = EventBrokerClient(config)
        self.tts_engine = TTSEngine(config)
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize the TTS Output Node."""
        logger.info("--- Initializing TTS Output Node ---")
        try:
            # Initialize TTS Engine
            if not self.tts_engine.initialize():
                logger.error("Failed to initialize TTS Engine. Cannot continue.")
                return False
            
            # Connect to L-EBM
            if not self.event_broker_client.connect():
                logger.error("Failed to connect to L-EBM. Cannot continue.")
                return False
            
            logger.info("--- TTS Output Node Initialization Complete ---")
            return True
            
        except Exception as e:
            logger.exception(f"Error initializing TTS Output Node: {e}")
            return False
    
    def start(self) -> bool:
        """Start the TTS Output Node."""
        if self.running:
            logger.warning("TTS Output Node already running.")
            return True
        
        try:
            logger.info("Starting TTS Output Node...")
            
            # Subscribe to speak request events
            if not self.event_broker_client.subscribe(self._process_speak_request):
                logger.error("Failed to subscribe to speak request events. Cannot continue.")
                return False
            
            self.running = True
            logger.info("TTS Output Node started successfully.")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting TTS Output Node: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the TTS Output Node."""
        if not self.running:
            logger.warning("TTS Output Node not running.")
            return
        
        logger.info("--- Stopping TTS Output Node ---")
        
        # Stop the TTS engine
        self.tts_engine.shutdown()
        
        # Disconnect from L-EBM
        self.event_broker_client.disconnect()
        
        self.running = False
        logger.info("--- TTS Output Node Stopped ---")
    
    def _process_speak_request(self, event: Dict[str, Any]) -> None:
        """Process a received speak request event."""
        try:
            # Extract necessary fields from the event
            event_id = event.get('event_id')
            text_to_speak = event.get('text_to_speak')
            correlation_id = event.get('correlation_id')
            user_id = event.get('user_id')
            speaking_rate = event.get('speaking_rate', self.config.tts_speed)
            priority = event.get('priority', 0)
            
            if not text_to_speak:
                logger.warning(f"Received speak request with no text: {event}")
                return
            
            logger.info(f"Processing speak request: '{text_to_speak[:50]}{'...' if len(text_to_speak) > 50 else ''}'")
            
            # Publish TTS start status
            start_status = {
                'status': 'start',
                'event_id': event_id,
                'correlation_id': correlation_id,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            self.event_broker_client.publish(self.config.tts_status_event_topic, start_status)
            
            # Queue the text for speaking
            speech_id = self.tts_engine.speak(text_to_speak, speaking_rate, priority)
            
            # Wait for speech to complete
            while self.tts_engine.speaking:
                time.sleep(0.1)
            
            # Publish TTS complete status
            complete_status = {
                'status': 'complete',
                'event_id': event_id,
                'speech_id': speech_id,
                'correlation_id': correlation_id,
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            self.event_broker_client.publish(self.config.tts_status_event_topic, complete_status)
            
        except Exception as e:
            logger.exception(f"Error processing speak request: {e}")


# --- Utility Functions ---
def list_audio_devices() -> None:
    """List available audio devices."""
    print("\n--- Available Audio Devices ---")
    devices = sd.query_devices()
    
    print("\nOutput Devices:")
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"  [{i}] {device['name']}")
    
    print("\nInput Devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{i}] {device['name']}")
    
    print("\nDefault Devices:")
    print(f"  Input: [{sd.default.device[0]}]")
    print(f"  Output: [{sd.default.device[1]}]")
    print("----------------------------\n")


def main() -> None:
    """Main function to parse arguments and run the TTS Output Node."""
    parser = argparse.ArgumentParser(description="AURA TTS Output Node")
    parser.add_argument("--config", type=str, default="config_tts.json", help="Path to configuration file")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--log-level", type=str, default="INFO", help="Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--file-log-level", type=str, default="INFO", help="File log level")
    parser.add_argument("--output-device", type=int, default=None, help="Override output device index")
    args = parser.parse_args()
    
    # Configure logging
    console_level_str = args.log_level.upper()
    console_level = getattr(logging, console_level_str, logging.INFO)
    log_stream_handler.setLevel(console_level)
    file_level_str = args.file_log_level.upper()
    file_level = getattr(logging, file_level_str, logging.INFO)
    root_logger.setLevel(min(file_level, console_level))
    logger.info(f"Console Log: {console_level_str}, File Log: {file_level_str}, Root Log: {min(file_level, console_level)}")
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Load configuration
    logger.info(f"Loading TTS Output Node config: {args.config}")
    config = Config()
    config.load_from_file(args.config)
    
    # Override with command-line arguments
    if args.output_device is not None:
        config.output_device_index = args.output_device
    
    # Initialize and run the TTS Output Node
    tts_node = None
    try:
        tts_node = TTSOutputNode(config)
        if not tts_node.initialize():
            logger.error("TTS Output Node initialization failed. Exiting.")
            return
        
        # Interactive selection of output device if not specified
        if config.output_device_index is None:
            list_audio_devices()
            try:
                device_idx = int(input("Enter output device number: ").strip())
                config.output_device_index = device_idx
                logger.info(f"TTS Using Output: {device_idx}")
            except ValueError:
                logger.info("Invalid device selection, using default output device.")
                config.output_device_index = sd.default.device[1]
            print("----------------------------\n")
        
        # Start the TTS Output Node
        if tts_node.start():
            logger.info("TTS Output Node is running. Press Ctrl+C to exit.")
            
            # Main loop - keep the program running and process events
            while True:
                time.sleep(1)
                
        else:
            logger.error("Failed to start TTS Output Node.")
            
    except KeyboardInterrupt:
        logger.info("TTS Output Node: Keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.exception(f"TTS Output Node: Critical error in main loop: {e}")
    finally:
        if tts_node:
            tts_node.stop()
    
    logger.info("TTS Output Node program finished.")


if __name__ == "__main__":
    main()
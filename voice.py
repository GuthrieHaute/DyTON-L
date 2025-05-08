import os
import sys
import time
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf # Keep for potential file transcription testing
import threading
import queue
import logging
import argparse
import json
import subprocess # Needed for espeak-ng check
import requests # Needed for Ollama
import datetime # AURA-DyTON: Added for ISO 8601 timestamp
from collections import deque
from dataclasses import dataclass, fields, asdict
from typing import Optional, List, Dict, Union, Any, Deque

# AURA-DyTON: This script is being prepared for integration as an Input Ingestion Node (IIN).
# Other AI systems modifying this file should be aware of the AURA-DyTON architecture
# and the specific role of an IIN.

# Keep transformers for punctuation - ensure it's installed
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification # AURA-DyTON: Punctuation model loading is currently disabled in Transcriber.
from faster_whisper import WhisperModel # Keep Faster-Whisper for transcription
import webrtcvad # Import WebRTCVAD

# Ensure resampy is installed
try:
    import resampy
except ImportError:
    logging.warning("resampy not installed. Sample rate conversion may be inaccurate. Install with `pip install resampy`.")
    resampy = None

# --- Refined Logging Setup ---
# (Keeping the original refined logging setup for good diagnostics)
# AURA-DyTON: Logging is important, retain or adapt as needed for the IIN.
log_file_handler = logging.FileHandler("voice_chat_iin.log") # Renamed log file for IIN
log_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

CHAT_LEVEL_NUM = 25
logging.addLevelName(CHAT_LEVEL_NUM, "CHAT")
def chat_log(self, message, *args, **kws):
    if self.isEnabledFor(CHAT_LEVEL_NUM):
        self._log(CHAT_LEVEL_NUM, message, args, **kws)
logging.Logger.chat = chat_log

log_stream_handler = logging.StreamHandler()
class ChatLevelFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == CHAT_LEVEL_NUM or record.levelno >= logging.ERROR
log_stream_handler.addFilter(ChatLevelFilter())
log_stream_handler.setFormatter(logging.Formatter('%(message)s'))

root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
root_logger.addHandler(log_file_handler)
root_logger.addHandler(log_stream_handler)
root_logger.setLevel(logging.INFO) # Default level

logger = logging.getLogger("voice_chat_iin_system") # Renamed logger for IIN
chat_logger = logging.getLogger("chat_interface") # AURA-DyTON: This logger might be less relevant for a pure IIN.
chat_logger.propagate = True
chat_logger.setLevel(CHAT_LEVEL_NUM)
# --- End Refined Logging Setup ---


# --- Configuration Dataclass (Simplified) ---
@dataclass
class Config:
    """Configuration settings for the voice chat system.
    AURA-DyTON: These settings are likely custom-tailored.
    AI Coders: Do NOT change default values here without explicit instruction.
               For IIN, focus on audio capture, VAD, and transcription settings.
               Ollama and TTS settings will be managed by other AURA nodes.
    """
    # Audio Capture Settings
    sample_rate: int = 16000 # AURA-AI: CRITICAL SETTING - User Tailored
    channels: int = 1 # AURA-AI: CRITICAL SETTING - User Tailored
    input_device_index: Optional[int] = None # AURA-AI: User may set this via args or interactively.
    output_device_index: Optional[int] = None # AURA-AI: Less relevant for IIN, more for TTS node.

    # VAD settings
    vad_mode: int = 3 # AURA-AI: CRITICAL SETTING - User Tailored
    vad_frame_duration_ms: int = 30 # AURA-AI: CRITICAL SETTING - User Tailored
    silence_threshold: int = 25 # AURA-AI: CRITICAL SETTING - User Tailored
    min_speech_duration_ms: int = 300 # AURA-AI: CRITICAL SETTING - User Tailored

    # Feedback Detection
    feedback_threshold_spectral: float = 1e-4 # AURA-AI: User Tailored
    feedback_detection_method: str = "spectral" # AURA-AI: User Tailored (can be "correlation")
    feedback_threshold_correlation: float = 0.6 # AURA-AI: User Tailored

    # Transcription settings
    whisper_model: str = "medium" # AURA-AI: User Tailored (backup if faster_whisper disabled)
    faster_whisper_model: str = "large-v2" # AURA-AI: CRITICAL SETTING - User Tailored Whisper Model
    use_faster_whisper: bool = True # AURA-AI: User Preference
    language: str = "en" # AURA-AI: User Tailored
    compute_type: str = "float16" # AURA-AI: CRITICAL SETTING - User Tailored (affects speed/accuracy/VRAM)
    faster_whisper_vad_filter: bool = True # AURA-AI: User Preference

    # Optional Features
    enable_noise_suppression: bool = False # AURA-AI: User Preference
    monitor_audio: bool = False # AURA-AI: User Preference (for debugging)

    # TTS Settings - AURA-DyTON: TTS is a separate node.
    # `tts_enabled` is retained as it's used by FeedbackDetector within the IIN
    # to determine if feedback detection logic should be active.
    # Other TTS-specific settings (voice path, TTS sample rate) are removed from IIN config.
    tts_enabled: bool = True # AURA-AI: User Preference for TTS. If true, FeedbackDetector logic is active.

    # Buffer Settings
    max_audio_buffer_size: int = 300 # AURA-AI: User Tailored

    # AURA-DyTON IIN Specific Settings
    event_broker_url: str = "redis://localhost:6379/0" # AURA-DyTON: Example L-EBM URL
    new_request_event_topic: str = "aura.requests.new" # AURA-DyTON: Example topic name

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Loads configuration from a JSON file, handling potential errors."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            field_names = {f.name for f in fields(cls)}
            filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
            instance = cls()
            for key, value in filtered_dict.items():
                setattr(instance, key, value)
            logger.info(f"Configuration successfully loaded from {config_path}")
            return instance
        except FileNotFoundError:
            logger.warning(f"Configuration file '{config_path}' not found. Using default settings.")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{config_path}': {e}. Using default settings.")
            return cls()
        except TypeError as e:
             logger.error(f"Type error loading config from '{config_path}': {e}. Using default settings.")
             return cls()
        except Exception as e:
            logger.error(f"Unexpected error loading config '{config_path}': {e}. Using default settings.")
            return cls()

    def to_file(self, config_path: str) -> None:
        """Saves the current configuration to a JSON file."""
        try:
            os.makedirs(os.path.dirname(config_path) or '.', exist_ok=True)
            config_dict = asdict(self)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to '{config_path}': {e}")
# --- End Configuration Dataclass ---


# --- AudioBuffer Class ---
# AURA-DyTON: This class is useful for the IIN. Retain.
class AudioBuffer:
    """Thread-safe buffer for audio chunks."""
    def __init__(self, max_size: int = 300): # AURA-AI: max_size is User Tailored via Config
        self.buffer = queue.Queue(maxsize=max_size)
        self.dropped_chunks = 0
    def put(self, audio_chunk: np.ndarray) -> None:
        try:
            if not isinstance(audio_chunk, np.ndarray):
                logger.warning(f"AudioBuffer received non-ndarray type: {type(audio_chunk)}. Attempting conversion.")
                try: audio_chunk = np.array(audio_chunk, dtype=np.float32)
                except Exception: logger.error("Failed to convert audio chunk to ndarray."); return
            self.buffer.put_nowait(audio_chunk.copy())
        except queue.Full:
            self.dropped_chunks += 1
            if self.dropped_chunks % 100 == 0:
                logger.warning(f"Audio buffer overflow, {self.dropped_chunks} chunks dropped so far.")
    def get(self) -> Optional[np.ndarray]:
        try:
            return self.buffer.get_nowait()
        except queue.Empty:
            return None
    def empty(self) -> bool:
        return self.buffer.empty()
    def size(self) -> int:
        return self.buffer.qsize()
# --- End AudioBuffer Class ---


# --- FeedbackDetector Class ---
# AURA-DyTON: Feedback detection is relevant for IIN to avoid transcribing its own output if co-located with TTS.
# However, in a fully decoupled DyTON-L, TTS output might not directly feed back into IIN's mic.
# AI Coders: Evaluate if this is strictly needed for the IIN or if it belongs with the TTS node.
# For now, assume it's useful for a robust IIN.
class FeedbackDetector:
    """Detects potential audio feedback by comparing input to recent output."""
    def __init__(self, config: Config): # AURA-AI: Config settings here are User Tailored
        chunk_duration_s = 0.250
        history_duration_s = 2.5
        chunk_size_samples = int(config.sample_rate * chunk_duration_s)
        max_chunks = int(history_duration_s / chunk_duration_s)
        self.recent_output_audio: Deque[np.ndarray] = deque(maxlen=max_chunks)
        self.output_lock = threading.Lock()
        self.config = config
        self.feedback_ignored_count = 0
        self._resampy_warning_logged = False

    def register_output(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """Registers played audio, resampling if necessary."""
        # AURA-DyTON: In an IIN, this method might not be called directly.
        # It would be called by the TTS node. If IIN needs to be aware of TTS output,
        # it might subscribe to a "TTSPlayedAudioEvent" from the L-EBM.
        if not self.config.tts_enabled: return # AURA-AI: User Preference
        target_sr = self.config.sample_rate
        if sample_rate != target_sr:
            if resampy:
                try:
                    if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
                    audio_data = resampy.resample(audio_data, sample_rate, target_sr)
                except Exception as e: logger.error(f"FeedbackDetector resample error: {e}"); return
            else:
                if not self._resampy_warning_logged:
                    logger.warning("FeedbackDetector: resampy unavailable for resampling."); self._resampy_warning_logged = True
                return

        if audio_data.ndim > 1: audio_data = np.mean(audio_data, axis=-1)
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)

        with self.output_lock:
            chunk_size = int(target_sr * 0.250)
            for i in range(0, len(audio_data), chunk_size):
                self.recent_output_audio.append(audio_data[i:i+chunk_size].copy())

    def detect_feedback(self, current_audio: np.ndarray) -> bool:
        """Compares current input audio chunk against recent output."""
        if not self.config.tts_enabled or not self.recent_output_audio: return False # AURA-AI: User Preference for TTS
        min_len = int(self.config.sample_rate * 0.05)
        if len(current_audio) < min_len: return False
        rms = np.sqrt(np.mean(current_audio**2))
        if rms < 1e-5: return False

        with self.output_lock:
            detected = False
            method = self.config.feedback_detection_method # AURA-AI: User Tailored
            try:
                if method == "correlation":
                    threshold = self.config.feedback_threshold_correlation # AURA-AI: User Tailored
                    len_c = len(current_audio)
                    for recent_chunk in reversed(self.recent_output_audio):
                        len_r = len(recent_chunk)
                        if len_r < len_c: continue
                        current_audio_f64 = current_audio.astype(np.float64)
                        recent_chunk_f64 = recent_chunk[:len_c].astype(np.float64)
                        cor = np.correlate(current_audio_f64, recent_chunk_f64, mode='valid')
                        norm = np.sqrt(np.sum(current_audio_f64**2) * np.sum(recent_chunk_f64**2))
                        if norm > 1e-9 and np.max(np.abs(cor)) / norm > threshold: detected = True; break
                elif method == "spectral":
                    threshold = self.config.feedback_threshold_spectral # AURA-AI: User Tailored
                    fft_c = np.fft.rfft(current_audio)
                    len_c_fft = len(fft_c)
                    for recent_chunk in reversed(self.recent_output_audio):
                        if len(recent_chunk) < len(current_audio): continue
                        fft_r = np.fft.rfft(recent_chunk[:len(current_audio)])
                        min_len_fft = min(len_c_fft, len(fft_r))
                        if min_len_fft == 0: continue
                        diff = np.mean(np.abs(fft_c[:min_len_fft] - fft_r[:min_len_fft]))
                        if diff < threshold: detected = True; break
                else: logger.warning(f"Unknown feedback detection method: {method}")
            except Exception as e: logger.error(f"Feedback detection error ({method}): {e}")

            if detected:
                self.feedback_ignored_count += 1
                if self.feedback_ignored_count % 50 == 1:
                    logger.info(f"Potential feedback detected ({method}). Ignoring input. (Count: {self.feedback_ignored_count})")
                return True
            return False
# --- End FeedbackDetector Class ---


# --- NoiseSupressor Class ---
# AURA-DyTON: Useful for IIN if enabled by user. Retain.
class NoiseSupressor:
    """Optional noise suppression using noisereduce library."""
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.enable_noise_suppression # AURA-AI: User Preference
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_duration_seconds = 2.5
        self.nr_module = None
        if self.enabled:
            try:
                import noisereduce as nr
                self.nr_module = nr
                logger.info("Noise suppression enabled. Waiting for initial silence to calibrate...")
            except ImportError:
                logger.warning("noisereduce library not installed. Noise suppression disabled. Run `pip install noisereduce`")
                self.enabled = False

    def calibrate(self, audio_data: np.ndarray) -> None:
        """Collects audio data during silence to build a noise profile."""
        if not self.enabled or self.is_calibrated or self.nr_module is None: return
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
        self.calibration_samples.append(audio_data)
        current_duration = sum(len(s) for s in self.calibration_samples) / self.config.sample_rate
        if current_duration >= self.calibration_duration_seconds:
            try:
                concatenated_samples = np.concatenate(self.calibration_samples)
                _ = self.nr_module.reduce_noise(y=concatenated_samples, sr=self.config.sample_rate) # Run once to cache profile
                self.is_calibrated = True
                self.calibration_samples = []
                logger.info(f"Noise suppression calibrated using {current_duration:.1f}s of audio.")
            except Exception as e:
                logger.error(f"Noise suppression calibration error: {e}"); self.calibration_samples = []; self.enabled = False
        elif len(self.calibration_samples) % 10 == 0:
             logger.info(f"Calibrating noise suppression... {current_duration:.1f}/{self.calibration_duration_seconds}s collected.")

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        """Applies noise reduction to the audio data."""
        if not self.enabled or not self.is_calibrated or self.nr_module is None: return audio_data
        if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
        try:
            return self.nr_module.reduce_noise(y=audio_data, sr=self.config.sample_rate, stationary=False, prop_decrease=0.8)
        except Exception as e: logger.error(f"Noise suppression processing error: {e}"); return audio_data
# --- End NoiseSupressor Class ---


# --- VADProcessor Class ---
# AURA-DyTON: Core component for IIN. Retain.
class VADProcessor:
    """Handles Voice Activity Detection using webrtcvad."""
    def __init__(self, config: Config): # AURA-AI: Config settings here are User Tailored
        self.config = config
        sr, fd = config.sample_rate, config.vad_frame_duration_ms
        if sr not in [8000, 16000, 32000, 48000]: raise ValueError(f"VAD unsupported sample rate: {sr}.")
        if fd not in [10, 20, 30]: raise ValueError(f"VAD unsupported frame duration: {fd} ms.")
        self.vad = webrtcvad.Vad(config.vad_mode)
        self.sample_rate = sr
        self.frame_duration_ms = fd
        self.frame_size = int(sr * fd / 1000)
        self.bytes_per_frame = self.frame_size * 2 # int16
        logger.info(f"VAD initialized (Mode: {config.vad_mode}, SR: {sr}, Frame: {fd}ms)")

    def is_speech(self, audio_frame_bytes: bytes) -> bool:
        """Checks if a raw audio frame (bytes, int16) contains speech."""
        if len(audio_frame_bytes) != self.bytes_per_frame:
            logger.warning(f"VAD received incorrect frame size: {len(audio_frame_bytes)} bytes, expected {self.bytes_per_frame}")
            return False
        try: return self.vad.is_speech(audio_frame_bytes, self.sample_rate)
        except Exception as e: logger.error(f"VAD is_speech error: {e}"); return False
# --- End VADProcessor Class ---


# --- Transcriber Class ---
# AURA-DyTON: Core component for IIN. Retain and adapt.
class Transcriber:
    """Handles audio transcription using Whisper or Faster-Whisper."""
    def __init__(self, config: Config): # AURA-AI: Config settings here are User Tailored
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.punctuation_tokenizer = None # AURA-AI: Punctuation currently disabled by user in original script.
        self.punctuation_model = None   # AURA-AI: Punctuation currently disabled by user in original script.
        self.whisper_module = None

    def load(self) -> None:
        """Loads the transcription and punctuation models."""
        # AURA-AI: Model choices and compute_type are CRITICAL User Tailored settings.
        model_type = 'Faster-Whisper' if self.config.use_faster_whisper else 'Whisper'
        model_name = self.config.faster_whisper_model if self.config.use_faster_whisper else self.config.whisper_model
        model_dir_name = "faster_whisper_models" if self.config.use_faster_whisper else "whisper_models"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        download_root = os.path.join(script_dir, model_dir_name)
        os.makedirs(download_root, exist_ok=True)
        logger.info(f"Loading {model_type} model: '{model_name}' (Device: {self.device}, Compute: {self.config.compute_type}, Download: {download_root})")
        try:
            if self.config.use_faster_whisper: # AURA-AI: User Preference
                self.model = WhisperModel(model_name, device=str(self.device), compute_type=self.config.compute_type, download_root=download_root)
            else:
                if self.whisper_module is None: import whisper; self.whisper_module = whisper
                self.model = self.whisper_module.load_model(model_name, device=self.device, download_root=download_root)
            logger.info(f"{model_type} model '{model_name}' loaded successfully.")

            # --- MODIFICATION START: Comment out punctuation loading ---
            # AURA-AI: Punctuation model loading was intentionally disabled in the original script.
            # AI Coders: Do NOT re-enable without explicit instruction.
            try:
                punct_model_name = "oliverguhr/fullstop-punctuation-multilang-large"
                logger.info(f"Attempting to load punctuation model: '{punct_model_name}' (CURRENTLY DISABLED BY USER)...")
                # self.punctuation_tokenizer = AutoTokenizer.from_pretrained(punct_model_name)
                # self.punctuation_model = AutoModelForTokenClassification.from_pretrained(punct_model_name)
                # self.punctuation_model.to(self.device)
                logger.info("Punctuation model load attempt skipped as per user's original script.")
                self.punctuation_model = None
                self.punctuation_tokenizer = None
            except Exception as e:
                logger.warning(f"Failed to load punctuation model (even if attempted): {e}. Punctuation will be disabled.")
                self.punctuation_model = None; self.punctuation_tokenizer = None
            # --- MODIFICATION END ---

        except Exception as e: logger.exception(f"Fatal error loading transcription model '{model_name}'"); raise

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribes an audio segment."""
        if self.model is None: logger.error("Transcription model not loaded."); return ""
        raw_transcription = ""
        try:
            target_sr = 16000 # Whisper preferred sample rate
            current_sr = self.config.sample_rate # AURA-AI: User Tailored sample rate
            if current_sr != target_sr:
                if resampy:
                    logger.debug(f"Resampling audio from {current_sr}Hz to {target_sr}Hz for transcription.")
                    if audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)
                    audio_data = resampy.resample(audio_data, current_sr, target_sr)
                else: logger.error("Cannot transcribe: Resampy unavailable for required resampling."); return ""
            elif audio_data.dtype != np.float32: audio_data = audio_data.astype(np.float32)

            if self.config.use_faster_whisper: # AURA-AI: User Preference
                segments, info = self.model.transcribe(audio_data, beam_size=5, language=self.config.language, vad_filter=self.config.faster_whisper_vad_filter) # AURA-AI: language and vad_filter are User Preferences
                raw_transcription = " ".join([seg.text for seg in segments])
                logger.debug(f"Faster-Whisper Info: Lang={info.language}(Prob={info.language_probability:.2f}), Duration={info.duration:.2f}s, VAD={self.config.faster_whisper_vad_filter}")
            else:
                if self.whisper_module is None: import whisper; self.whisper_module = whisper
                result = self.model.transcribe(audio_data, language=self.config.language, fp16=torch.cuda.is_available()) # AURA-AI: language is User Preference
                raw_transcription = result["text"]
            logger.debug(f"Raw transcription: {raw_transcription}")
            if self.punctuation_model and self.punctuation_tokenizer and raw_transcription.strip(): # AURA-AI: Punctuation currently disabled
                return self.add_punctuation(raw_transcription).strip()
            else:
                logger.debug("Punctuation disabled or model not loaded, returning raw transcription.")
                return raw_transcription.strip()
        except Exception as e: logger.exception("Transcription error occurred"); return raw_transcription.strip()

    def add_punctuation(self, text: str) -> str:
        """Adds punctuation to text using the loaded model."""
        # AURA-AI: Punctuation currently disabled. This function will likely not be called.
        if not text.strip() or not self.punctuation_model or not self.punctuation_tokenizer:
            logger.debug("Skipping punctuation addition (model not loaded or text empty).")
            return text
        logger.debug("Applying punctuation model...")
        try:
            punct_pipeline = pipeline("token-classification", model=self.punctuation_model, tokenizer=self.punctuation_tokenizer, device=self.device, aggregation_strategy="simple")
            result = punct_pipeline(text)
            punctuated_text = ""
            last_end = 0
            for entity in result:
                start, end = entity['start'], entity['end']
                punctuated_text += text[last_end:start] + entity['word']
                label = entity['entity_group']
                if label == "PERIOD": punctuated_text += "."
                elif label == "COMMA": punctuated_text += ","
                elif label == "QUESTION": punctuated_text += "?"
                last_end = end
            punctuated_text += text[last_end:]
            punctuated_text = punctuated_text.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").strip()
            logger.debug(f"Punctuated text: {punctuated_text[:150]}...")
            return punctuated_text
        except Exception as e: logger.error(f"Punctuation application error: {e}"); return text
# --- End Transcriber Class ---


# --- L-EBM Client (Placeholder) ---
# AURA-DyTON: This class will handle communication with the Lightweight Event Broker Middleware.
class EventBrokerClient:
    """Handles connection and publishing to the L-EBM (e.g., Redis)."""
    def __init__(self, config: Config):
        self.config = config
        self.client = None # Placeholder for actual client (e.g., redis.Redis)
        self.is_connected = False
        self.max_retries = 3
        self.retry_delay_seconds = 5
        # For a real Redis client, you'd import redis and initialize here or in connect()
        # import redis # Example
        # self.client = redis.Redis.from_url(self.config.event_broker_url) # Example

    def connect(self) -> bool:
        """Establishes connection to the L-EBM with retries."""
        for attempt in range(self.max_retries):
            logger.info(f"Attempting to connect to L-EBM at {self.config.event_broker_url} (Attempt {attempt + 1}/{self.max_retries})...")
            try:
                # Simulate connection attempt
                # In a real implementation, this would involve self.client.ping() or similar
                # For example, if using Redis:
                # if self.client.ping():
                #    logger.info(f"Successfully connected to L-EBM (Topic: {self.config.new_request_event_topic}).")
                #    self.is_connected = True
                #    return True
                # else:
                #    logger.warning(f"L-EBM connection attempt {attempt + 1} failed: Ping unsuccessful.")
                #    if attempt < self.max_retries - 1: time.sleep(self.retry_delay_seconds)
                #    continue
                logger.info(f"L-EBM connection simulated for {self.config.event_broker_url} (Topic: {self.config.new_request_event_topic}).")
                self.is_connected = True # Assume connection is successful for simulation
                return True
            except Exception as e:
                logger.error(f"L-EBM connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay_seconds} seconds...")
                    time.sleep(self.retry_delay_seconds)
                else:
                    logger.error(f"All {self.max_retries} L-EBM connection attempts failed.")
                    self.is_connected = False
                    return False
        self.is_connected = False
        return False

    def publish(self, topic: str, event_payload: Dict[str, Any]) -> bool:
        """Publishes an event to the specified topic on the L-EBM with retries."""
        if not self.is_connected:
            logger.warning("L-EBM client not connected. Attempting to reconnect before publishing...")
            if not self.connect():
                logger.error("L-EBM reconnection failed. Cannot publish event.")
                # AURA-DyTON: Consider a dead-letter queue mechanism here for critical events
                return False

        for attempt in range(self.max_retries):
            try:
                payload_str = json.dumps(event_payload)
                # Simulate publishing
                # In a real implementation, this would be:
                # self.client.publish(topic, payload_str)
                logger.info(f"L-EBM SIMULATED PUBLISH to topic \'{topic}\' (Attempt {attempt+1}): {payload_str}")
                return True
            except json.JSONDecodeError as ser_e:
                logger.error(f"Event serialization error: {ser_e}. Cannot publish malformed event.")
                return False # Do not retry on serialization error
            except Exception as e:
                logger.error(f"Failed to publish event to L-EBM topic \'{topic}\' (Attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying publish in {self.retry_delay_seconds} seconds...")
                    time.sleep(self.retry_delay_seconds)
                else:
                    logger.error(f"All {self.max_retries} L-EBM publish attempts failed for topic \'{topic}\'.")
                    # AURA-DyTON: Consider a dead-letter queue mechanism here
                    return False
        return False

    def disconnect(self) -> None:
        """Disconnects from the L-EBM."""
        if self.is_connected:
            logger.info("Disconnecting from L-EBM...")
            # In a real implementation, close the client connection if necessary
            # self.client.close() # Example for some clients
            self.is_connected = False
            logger.info("Disconnected from L-EBM (simulated).")
# --- End L-EBM Client ---


# --- InputIngestionNode Class (Simplified) ---
# AURA-DyTON: This class will be significantly refactored for the IIN.
# It will primarily manage audio capture, VAD, transcription, and event publishing.
class InputIngestionNode:
    """Main system orchestrator for audio capture, VAD, transcription, and event publishing for AURA-DyTON IIN."""
    def __init__(self, config: Config):
        self.config = config

        self.audio_buffer = AudioBuffer(config.max_audio_buffer_size)
        self.feedback_detector = FeedbackDetector(config) # AURA-DyTON: Re-evaluate need or adapt for event-driven feedback info
        self.noise_suppressor = NoiseSupressor(config)
        self.vad_processor = VADProcessor(config)
        self.transcriber = Transcriber(config)

        self.recording = False
        self.finished = False
        self.capture_thread = None
        self.vad_process_thread = None
        self._frame_buffer = b''
        self._speech_frames: Deque[bytes] = deque()
        self._silence_count = 0
        self._is_speech_active = False

        # AURA-DyTON: Placeholder for TTS activity state, to be updated via L-EBM events.
        self.external_tts_active = False

        # AURA-DyTON: Add L-EBM client for publishing events
        self.event_broker_client = EventBrokerClient(config)

    def initialize(self) -> None:
        """Initializes system components for IIN."""
        logger.info("--- Initializing Input Ingestion Node Components ---")
        cuda_ok = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_ok}" + (f" (Device: {torch.cuda.get_device_name(0)})" if cuda_ok else ""))
        try:
            self.transcriber.load()
            logger.info(f"Noise Suppression: {'Enabled' if self.config.enable_noise_suppression else 'Disabled'}")
            # AURA-DyTON: Initialize L-EBM client here
            self.event_broker_client.connect()
            logger.info("--- Input Ingestion Node Initialization Complete ---")
        except Exception as e: logger.exception("Fatal error during IIN initialization."); raise

    def audio_capture_thread(self) -> None:
        """Thread to capture audio from the input device."""
        block_size = self.vad_processor.frame_size * 15
        stream = None
        try:
            logger.info(f"Starting audio input (Device: {self.config.input_device_index}, SR: {self.config.sample_rate}, Block: {block_size})") # AURA-AI: input_device_index, sample_rate are User Tailored
            stream = sd.InputStream(samplerate=self.config.sample_rate, channels=self.config.channels, dtype='float32', blocksize=block_size, device=self.config.input_device_index)
            logger.info(f"Audio stream opened. Latency: {stream.latency:.4f}s")
            with stream:
                logger.info("Audio capture started. Listening...")
                while self.recording:
                    try:
                        audio_chunk, status = stream.read(block_size)
                        if status: logger.warning(f"Audio stream status: {status}")
                        if audio_chunk.ndim > 1: audio_chunk = np.mean(audio_chunk, axis=-1)
                        if audio_chunk.dtype != np.float32: audio_chunk = audio_chunk.astype(np.float32)
                        self.audio_buffer.put(audio_chunk)
                    except sd.PortAudioError as pae: logger.error(f"Audio capture PortAudioError: {pae}. Stopping."); self.recording = False; break
                    except Exception as loop_e: logger.exception("Unexpected error in audio capture loop"); time.sleep(0.1)
        except sd.PortAudioError as pae: logger.error(f"Failed to open audio input stream: {pae}."); self.finished = True
        except ValueError as ve: logger.error(f"Audio stream config error: {ve}."); self.finished = True
        except Exception as e: logger.exception("Audio capture setup failed"); self.finished = True
        finally:
            if stream and not stream.closed:
                try: stream.close(); logger.info("Audio input stream closed.")
                except Exception as ce: logger.error(f"Error closing audio input stream: {ce}")
            self.finished = True; logger.info("Audio capture thread terminated.")

    def vad_processing_thread(self) -> None:
        """Thread to process audio from buffer, apply VAD, and trigger transcription for event publishing."""
        frame_bytes_len = self.vad_processor.bytes_per_frame
        logger.info("VAD processing thread started.")
        while True:
            buffer_empty = self.audio_buffer.empty()
            frame_buffer_empty = (len(self._frame_buffer) == 0)
            if not self.recording and self.finished and buffer_empty and frame_buffer_empty: break

            try:
                chunk = self.audio_buffer.get()
                if chunk is None:
                    if self.recording or not self.finished: time.sleep(0.005)
                    continue

                proc_chunk = chunk
                if self.config.enable_noise_suppression: # AURA-AI: User Preference
                    if self.noise_suppressor.is_calibrated: proc_chunk = self.noise_suppressor.process(proc_chunk)
                    else: self.noise_suppressor.calibrate(proc_chunk); continue

                if proc_chunk.ndim > 1: proc_chunk = np.mean(proc_chunk, axis=-1)
                proc_chunk_clamped = np.clip(proc_chunk, -1.0, 1.0)
                chunk_bytes = (proc_chunk_clamped * 32767).astype(np.int16).tobytes()
                self._frame_buffer += chunk_bytes

                while len(self._frame_buffer) >= frame_bytes_len:
                    frame = self._frame_buffer[:frame_bytes_len]
                    self._frame_buffer = self._frame_buffer[frame_bytes_len:]

                    # AURA-DyTON: Feedback detection logic might need to change if TTS is a separate node.
                    # For now, IIN relies on external_tts_active flag, to be set by L-EBM event handling.
                    is_tts_speaking = self.external_tts_active
                    if is_tts_speaking:
                        if self._is_speech_active: logger.debug("TTS speaking, discarding speech frames."); self._speech_frames.clear()
                        self._is_speech_active = False; self._silence_count = 0; continue

                    is_speech = self.vad_processor.is_speech(frame)
                    frame_f32 = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32767.0
                    is_feedback = self.feedback_detector.detect_feedback(frame_f32)

                    if is_speech and not is_feedback:
                        self._speech_frames.append(frame); self._silence_count = 0
                        if not self._is_speech_active: logger.debug("VAD: Speech started.")
                        self._is_speech_active = True
                    elif is_speech and is_feedback: self._silence_count = 0
                    else:
                        if self._is_speech_active:
                            self._silence_count += 1
                            if self._silence_count >= self.config.silence_threshold: # AURA-AI: User Tailored
                                logger.debug(f"VAD: Speech ended (Silence >= {self.config.silence_threshold}).")
                                if len(self._speech_frames) > 0:
                                    audio_bytes = b''.join(self._speech_frames)
                                    self._speech_frames.clear()
                                    audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                                    segment_duration_ms = (len(audio_f32) / self.config.sample_rate) * 1000
                                    if segment_duration_ms >= self.config.min_speech_duration_ms: # AURA-AI: User Tailored
                                        self._transcribe_and_publish_event(audio_f32)
                                    else: logger.info(f"Speech segment too short ({segment_duration_ms:.0f}ms), ignoring.")
                                else: logger.debug("Speech ended but buffer empty.")
                                self._silence_count = 0; self._is_speech_active = False
            except Exception as e: logger.exception("Error in VAD processing loop"); time.sleep(0.01)

        if len(self._speech_frames) > 0 and not self.external_tts_active:
            logger.info("Processing remaining speech frames...")
            try:
                audio_bytes = b''.join(self._speech_frames)
                audio_f32 = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                segment_duration_ms = (len(audio_f32) / self.config.sample_rate) * 1000
                if segment_duration_ms >= self.config.min_speech_duration_ms: # AURA-AI: User Tailored
                    self._transcribe_and_publish_event(audio_f32)
                else: logger.info(f"Final speech segment too short ({segment_duration_ms:.0f}ms), ignoring.")
            except Exception as e: logger.exception("Error processing final speech frames")
            finally: self._speech_frames.clear()
        logger.info("VAD processing thread terminated.")

    def _transcribe_and_publish_event(self, audio_segment: np.ndarray) -> None:
        """Transcribes a speech segment and publishes a NewRequestEvent."""
        start_time = time.time()
        segment_duration_sec = len(audio_segment) / self.config.sample_rate
        logger.info(f"--- IIN: Processing speech segment for event ({segment_duration_sec:.2f}s) ---")
        try:
            if audio_segment.size == 0: logger.warning("IIN: Empty audio segment."); return
            if self.config.monitor_audio: # AURA-AI: User Preference
                try: logger.debug("IIN: Playing back captured audio..."); sd.play(audio_segment, self.config.sample_rate, device=self.config.output_device_index) # AURA-AI: User Tailored output device
                except Exception as e: logger.warning(f"IIN: Audio monitoring failed: {e}")

            logger.debug("IIN: Transcribing...")
            transcription = self.transcriber.transcribe(audio_segment)
            if not transcription or not transcription.strip(): logger.info("IIN: Empty transcription."); return

            logger.info(f"IIN: Transcription: '{transcription}'")

            event_payload = {
                "request_id": f"req_{int(time.time()*1000)}_{np.random.randint(1000,9999)}", # Simple unique ID
                "user_id": "default_user", # AURA-DyTON: Placeholder, enhance with actual user ID if available
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), # AURA-DyTON: Updated to ISO 8601
                "input_type": "voice",
                "transcribed_text": transcription,
            }
            logger.info(f"IIN: Publishing NewRequestEvent: {event_payload}")
            if self.event_broker_client:
                try:
                    self.event_broker_client.publish(self.config.new_request_event_topic, event_payload)
                    logger.debug(f"IIN: Event published to topic \'{self.config.new_request_event_topic}\'.")
                except Exception as pub_e:
                    logger.error(f"IIN: Failed to publish event: {pub_e}")
            else:
                logger.warning("IIN: Event broker client not initialized. Cannot publish event.")

        except Exception as e:
            logger.exception("IIN: Error processing speech segment for event")
        finally:
            end_time = time.time(); latency = end_time - start_time
            logger.info(f"--- IIN: Finished processing segment (Total time: {latency:.2f}s) ---")

    def start(self) -> None:
        """Starts the IIN audio capture and processing threads."""
        try:
            self.initialize()
            self.recording = True; self.finished = False
            logger.info("IIN: Starting audio capture and VAD threads...")
            self.capture_thread = threading.Thread(target=self.audio_capture_thread, daemon=True)
            self.vad_process_thread = threading.Thread(target=self.vad_processing_thread, daemon=True)
            self.capture_thread.start()
            time.sleep(0.3)
            self.vad_process_thread.start()
            logger.info("IIN System started. Listening...")
        except Exception as e: logger.exception("IIN System startup failed"); self.stop(); raise

    def stop(self) -> None:
        """Stops all IIN threads and cleans up resources."""
        logger.info("--- IIN: Initiating System Shutdown ---")
        self.recording = False

        logger.info("IIN: Waiting for audio capture thread...")
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive(): logger.warning("IIN: Capture thread join timeout.")
            else: logger.info("IIN: Capture thread stopped.")
        else: logger.debug("IIN: Capture thread not running/already stopped.")

        logger.info("IIN: Waiting for VAD processing thread...")
        if self.vad_process_thread and self.vad_process_thread.is_alive():
            self.vad_process_thread.join(timeout=10.0) # Increased timeout for VAD thread
            if self.vad_process_thread.is_alive(): logger.warning("IIN: VAD thread join timeout.")
            else: logger.info("IIN: VAD thread stopped.")
        else: logger.debug("IIN: VAD thread not running/already stopped.")
        
        if self.event_broker_client:
            self.event_broker_client.disconnect()
        logger.info("--- IIN: System Shutdown Complete ---")

    def transcribe_file(self, file_path: str) -> str:
        """Transcribes a single audio file (Kept for potential testing)."""
        logger.info(f"--- Starting File Transcription: {file_path} ---")
        transcription = ""
        try:
            if self.transcriber.model is None: logger.info("Loading model..."); self.transcriber.load()
            if not os.path.exists(file_path): logger.error(f"File not found: {file_path}"); return ""
            logger.debug(f"Loading audio: {file_path}")
            audio, sr = sf.read(file_path, dtype='float32')
            logger.info(f"File loaded. SR: {sr} Hz, Duration: {len(audio)/sr:.2f}s, Shape: {audio.shape}")
            if audio.ndim > 1: logger.debug("Converting to mono."); audio = np.mean(audio, axis=-1)
            target_sr = 16000
            if sr != target_sr:
                logger.info(f"Resampling {sr}Hz to {target_sr}Hz...")
                if resampy: audio = resampy.resample(audio, sr, target_sr); logger.debug("Resampling complete.")
                else: logger.error("Resampy unavailable."); return ""
            logger.info("Starting transcription...")
            t_start = time.time()
            transcription = self.transcriber.transcribe(audio)
            t_end = time.time()
            logger.info(f"File transcription complete ({t_end - t_start:.2f}s).")
            log_trans = (transcription[:500] + '...') if len(transcription) > 500 else transcription
            logger.info(f"Result Snippet:\n{log_trans}")
            return transcription
        except Exception as e: logger.exception(f"Error transcribing file: {file_path}"); return ""
        finally: logger.info(f"--- Finished File Transcription: {file_path} ---")
# --- End InputIngestionNode Class ---


# --- Utility Functions ---
def list_audio_devices():
    """Prints available audio devices and their indices."""
    print("\n--- Available Audio Devices ---")
    try:
        devices = sd.query_devices()
        default_in, default_out = sd.default.device
        print(f"Default Input: {default_in}, Default Output: {default_out}\n")
        header = f"{'Idx':<3} | {'Name':<35} | {'Host API':<14} | {'In':<3} | {'Out':<3} | {'Def SR':<10}"
        print(header); print("-" * len(header))
        for i, d in enumerate(devices):
            name = d.get('name', 'N/A')[:35]
            host = d.get('hostapi_name', 'N/A')[:14]
            in_ch = d.get('max_input_channels', '?')
            out_ch = d.get('max_output_channels', '?')
            sr = d.get('default_samplerate', 0.0)
            in_def = "[IN]" if i == default_in else ""
            out_def = "[OUT]" if i == default_out else ""
            print(f"{i:<3} | {name:<35} | {host:<14} | {in_ch:<3} | {out_ch:<3} | {sr:<10.0f} {in_def}{out_def}")
    except Exception as e: logger.error(f"Error querying audio devices: {e}"); print("\nCould not list devices.")
    print("-" * len(header)); print("-----------------------------\n")
# --- End Utility Functions ---


# --- Main Execution (Simplified for IIN) ---
def main() -> None:
    """Main function to parse arguments and run the IIN system."""
    parser = argparse.ArgumentParser(description="AURA Input Ingestion Node (Voice).", formatter_class=argparse.ArgumentDefaultsHelpFormatter) # AURA-DyTON: Updated description
    parser.add_argument("--config", type=str, default="config_iin.json", help="Path to the IIN configuration file.") # AURA-DyTON: Updated default config file name
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "CHAT", "WARNING", "ERROR", "CRITICAL"], help="Console log level.")
    parser.add_argument("--file-log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="File log level.")
    parser.add_argument("--input-device", type=int, default=None, help="Override input device index.")
    parser.add_argument("--output-device", type=int, default=None, help="Override output device index (for audio monitoring).")
    parser.add_argument("--file", type=str, help="Transcribe a specific audio file (for testing Transcriber) and exit.")
    args = parser.parse_args()

    # Configure Logging (same as before, ensure levels are appropriate for IIN)
    console_level_str = args.log_level.upper()
    console_level = CHAT_LEVEL_NUM if console_level_str == "CHAT" else getattr(logging, console_level_str, logging.INFO)
    log_stream_handler.setLevel(console_level)
    file_level_str = args.file_log_level.upper()
    file_level = getattr(logging, file_level_str, logging.INFO)
    root_min_level = min(file_level, console_level if console_level != CHAT_LEVEL_NUM else logging.INFO)
    if console_level == CHAT_LEVEL_NUM: root_min_level = min(root_min_level, CHAT_LEVEL_NUM)
    root_logger.setLevel(root_min_level)
    logger.info(f"Console Log: {logging.getLevelName(console_level)}, File Log: {logging.getLevelName(file_level)}, Root Log: {logging.getLevelName(root_min_level)}")


    if args.list_devices: list_audio_devices(); return

    config = Config()
    config_path = args.config
    if os.path.exists(config_path): logger.info(f"Loading IIN config: {config_path}"); config = Config.from_file(config_path)
    else: logger.warning(f"IIN Config '{config_path}' not found. Using defaults and saving."); config.to_file(config_path) # Save defaults if not found

    if args.input_device is not None: logger.info(f"Overriding input device: {args.input_device}"); config.input_device_index = args.input_device
    if args.output_device is not None: logger.info(f"Overriding output device: {args.output_device}"); config.output_device_index = args.output_device

    system_iin: Optional[InputIngestionNode] = None
    try:
        system_iin = InputIngestionNode(config)
    except Exception as init_e:
        logger.exception("IIN System initialization failed."); print(f"\nFATAL ERROR: {init_e}"); return

    if args.file:
        logger.info(f"--- IIN: Running File Transcription Test Mode ---")
        if not os.path.exists(args.file): logger.error(f"Input file not found: {args.file}"); print(f"ERROR: Not found: {args.file}"); return
        transcription_result = system_iin.transcribe_file(args.file)
        print("\n--- Transcription Result ---"); print(transcription_result); print("--------------------------\n")
    else:
        logger.info(f"--- IIN: Running Real-Time Mode ---")
        try:
            if config.input_device_index is None: # AURA-AI: User preference for device selection
                list_audio_devices()
                print("\n--- Audio Device Selection (IIN) ---")
                try:
                    default_in, _ = sd.default.device # IIN primarily needs input
                    in_idx = input(f"Enter INPUT device index for IIN (default {default_in}): ").strip()
                    config.input_device_index = int(in_idx) if in_idx else default_in
                    if config.monitor_audio and config.output_device_index is None:
                        _, default_out = sd.default.device
                        out_idx = input(f"Enter OUTPUT device index for IIN audio monitoring (default {default_out}): ").strip()
                        config.output_device_index = int(out_idx) if out_idx else default_out
                    logger.info(f"IIN Using Input: {config.input_device_index}" + (f", Output (monitor): {config.output_device_index}" if config.monitor_audio else ""))
                except ValueError: logger.error("Invalid device index."); print("Invalid number."); return
                except Exception as e: logger.exception("Device selection error."); config.input_device_index = sd.default.device[0]
                print("----------------------------\n")

            system_iin.start()
            logger.info("IIN: Voice input node activated. Listening for speech to publish events...")
            while True:
                try:
                    time.sleep(1)
                    if not system_iin.capture_thread or not system_iin.capture_thread.is_alive():
                        logger.critical("IIN CRITICAL: Capture thread stopped! Shutting down."); break
                    if not system_iin.vad_process_thread or not system_iin.vad_process_thread.is_alive():
                        logger.critical("IIN CRITICAL: VAD thread stopped! Shutting down."); break
                except EOFError: logger.info("IIN: Input stream closed (EOF). Shutting down."); break
        except KeyboardInterrupt: logger.info("IIN: Keyboard interrupt. Shutting down...")
        except Exception as e: logger.exception("IIN: Critical error in main loop")
        finally:
            if system_iin: system_iin.stop()

    logger.info("IIN Program finished.")

if __name__ == "__main__":
    main()

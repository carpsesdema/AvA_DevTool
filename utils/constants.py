# utils/constants.py
import logging
import os
import sys

logger = logging.getLogger(__name__)

APP_NAME = "AvA: PySide6 Rebuild"
APP_VERSION = "0.1.0-Phase1"

# --- Backend Configuration ---
DEFAULT_CHAT_BACKEND_ID = "ollama_chat_default" # CORRECTED: Changed default to Ollama
DEFAULT_GEMINI_CHAT_MODEL = "gemini-1.5-flash-latest" # Keep for Gemini still
DEFAULT_OLLAMA_CHAT_MODEL = "llama2:13b" # ADDED: New default for Ollama

# --- UI Aesthetics ---
CHAT_FONT_FAMILY = "Segoe UI"
CHAT_FONT_SIZE = 10
LOADING_GIF_FILENAME = "loading.gif"
APP_ICON_FILENAME = "Synchat.ico" # Or "ava_logo.svg" if you add it

USER_BUBBLE_COLOR_HEX = "#0B71E6"
USER_TEXT_COLOR_HEX = "#FFFFFF"
AI_BUBBLE_COLOR_HEX = "#3E3E3E"
AI_TEXT_COLOR_HEX = "#E0E0E0"
SYSTEM_BUBBLE_COLOR_HEX = "#5A5A5A"
SYSTEM_TEXT_COLOR_HEX = "#B0B0B0"
ERROR_BUBBLE_COLOR_HEX = "#730202"
ERROR_TEXT_COLOR_HEX = "#FFCCCC"
BUBBLE_BORDER_COLOR_HEX = "#2D2D2D"
TIMESTAMP_COLOR_HEX = "#888888"
CODE_BLOCK_BG_COLOR_HEX = "#1E1E1E"

# --- Paths ---
if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(sys.executable)
else:
    APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

USER_DATA_DIR_NAME = ".ava_pys6_data_p1"
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), USER_DATA_DIR_NAME)

ASSETS_DIR_NAME = "assets"
ASSETS_PATH = os.path.join(APP_BASE_DIR, ASSETS_DIR_NAME)

STYLESHEET_FILENAME = "style.qss"
BUBBLE_STYLESHEET_FILENAME = "bubble_style.qss"
UI_DIR_NAME = "ui"
UI_DIR_PATH = os.path.join(APP_BASE_DIR, UI_DIR_NAME)

STYLE_PATHS_TO_CHECK = [
    os.path.join(APP_BASE_DIR, UI_DIR_NAME, STYLESHEET_FILENAME),
    os.path.join(APP_BASE_DIR, STYLESHEET_FILENAME)
]
BUBBLE_STYLESHEET_PATH = os.path.join(UI_DIR_PATH, BUBBLE_STYLESHEET_FILENAME)

# --- Logging ---
LOG_LEVEL = "DEBUG"
LOG_FILE_NAME = "ava_pys6_phase1.log"
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    logger.info(f"User data directory ensured at: {USER_DATA_DIR}")
except OSError as e:
    logger.critical(f"CRITICAL: Error creating user data directory in constants.py: {e}", exc_info=True)
# utils/constants.py
import logging
import os
import sys

logger = logging.getLogger(__name__)

APP_NAME = "AvA: PySide6 Rebuild"
APP_VERSION = "0.1.0-Phase1"


DEFAULT_CHAT_BACKEND_ID = "ollama_chat_default"
DEFAULT_GEMINI_CHAT_MODEL = "gemini-2.0-flash-latest"
DEFAULT_OLLAMA_CHAT_MODEL = "llama3:latest"


GENERATOR_BACKEND_ID = "ollama_generator_default"
DEFAULT_OLLAMA_GENERATOR_MODEL = "codellama:13b"

CODER_AI_SYSTEM_PROMPT = """You are an expert Python code generation assistant. Your primary goal is to produce exceptionally clean, correct, and robust Python code.
You will be given a task to generate or update a specific file. You MUST strictly adhere to the provided detailed instructions and any original file content if updating.

**Key Requirements for Your Output:**
1.  **Accuracy & Completeness:**
    * Precisely implement all logic and features described in the instructions.
    * **If updating an existing file:** Meticulously preserve all unchanged original code. Only modify the specified sections. Do NOT omit any original code unless explicitly instructed to remove it.
    * If generating a new file, ensure all necessary components (imports, functions, classes, etc.) are included.
2.  **Code Quality & Python Best Practices:**
    * Write idiomatic Python, leveraging built-in functions and standard library features effectively.
    * Strictly follow PEP 8 style guidelines (e.g., line length around 99 characters, clear naming conventions).
    * Include comprehensive type hints (PEP 484) for all function/method signatures and important variables.
    * Write clear, concise, and informative docstrings (PEP 257) for all modules, classes, functions, and methods, explaining purpose, arguments, and returns.
    * Add inline comments for any complex, non-obvious, or critical sections of logic.
    * Ensure the code is robust. Consider potential edge cases and include error handling (e.g., try-except blocks) where appropriate, especially for I/O operations or external API calls. Aim for graceful failure.
    * Strive for modular functions and classes that adhere to the Single Responsibility Principle where feasible.
    * Write efficient code, but prioritize clarity and maintainability unless performance is explicitly stated as a critical requirement for a specific part of the code.
3.  **Output Format:**
    * Your *entire* response MUST be a single Markdown Python code block.
    * The code block must start with ```python path/to/filename.ext\\n (replace path/to/filename.ext with the actual relative file path provided in the instructions).
    * The code block must end with ```.
    * There should be NO other text, explanations, apologies, or conversational filler before or after this single code block.
4.  **Self-Correction & Pitfall Avoidance:**
    * Before finalizing your response, critically review your generated code against all instructions and the Python best practices outlined above.
    * Avoid placeholder comments like `# TODO` or `# Implement later` unless specifically part of the instructions. Deliver complete, working code for the requested scope.
    * Ensure all necessary imports are included at the beginning of the file.

Produce the most clean, readable, maintainable, and correct Python code possible for the given task.
"""


CHAT_FONT_FAMILY = "Segoe UI"
CHAT_FONT_SIZE = 10
LOADING_GIF_FILENAME = "loading.gif"
APP_ICON_FILENAME = "Synchat.ico"

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

LOG_LEVEL = "DEBUG"
LOG_FILE_NAME = "ava_pys6_phase1.log"
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    logger.info(f"User data directory ensured at: {USER_DATA_DIR}")
except OSError as e:
    logger.critical(f"CRITICAL: Error creating user data directory in constants.py: {e}", exc_info=True)
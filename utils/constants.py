# utils/constants.py
import logging
import os
import sys

logger = logging.getLogger(__name__)

APP_NAME = "AvA: PySide6 Rebuild"
APP_VERSION = "1.0.5"


# --- LLM Configuration ---
# AVA_ASSISTANT_MODIFIED: Changed DEFAULT_CHAT_BACKEND_ID to be a consistent adapter key
DEFAULT_CHAT_BACKEND_ID = "gemini_chat_default"
DEFAULT_GEMINI_CHAT_MODEL = "models/gemini-2.5-flash-preview-05-20" # The actual model for the gemini_chat_default adapter
DEFAULT_OLLAMA_CHAT_MODEL = "llama3:latest" # The actual model for the ollama_chat_default adapter
# DEFAULT_GPT_CHAT_MODEL = "gpt-3.5-turbo" # Example if you add a GPT default

GENERATOR_BACKEND_ID = "ollama_generator_default" # This is an adapter key
DEFAULT_OLLAMA_GENERATOR_MODEL = "codellama:13b" # The actual model for this adapter

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


# --- UI & Asset Constants ---
CHAT_FONT_FAMILY = "Segoe UI"
CHAT_FONT_SIZE = 12
LOADING_GIF_FILENAME = "loading.gif"
APP_ICON_FILENAME = "Synchat.ico"

USER_BUBBLE_COLOR_HEX = "#00e676"
USER_TEXT_COLOR_HEX = "#0d1117"
AI_BUBBLE_COLOR_HEX = "#21262d"
AI_TEXT_COLOR_HEX = "#f0f6fc"
SYSTEM_BUBBLE_COLOR_HEX = "#30363d"
SYSTEM_TEXT_COLOR_HEX = "#c9d1d9"
ERROR_BUBBLE_COLOR_HEX = "#f85149"
ERROR_TEXT_COLOR_HEX = "#ffffff"
BUBBLE_BORDER_COLOR_HEX = "#30363d"
TIMESTAMP_COLOR_HEX = "#6e7681"
CODE_BLOCK_BG_COLOR_HEX = "#161b22"

# --- Path Constants ---
if getattr(sys, 'frozen', False):
    APP_BASE_DIR = os.path.dirname(sys.executable)
else:
    APP_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root

USER_DATA_DIR_NAME = ".ava_pys6_data_p1"
USER_DATA_DIR = os.path.join(os.path.expanduser("~"), USER_DATA_DIR_NAME)

ASSETS_DIR_NAME = "assets"
ASSETS_PATH = os.path.join(APP_BASE_DIR, ASSETS_DIR_NAME)

STYLESHEET_FILENAME = "style.qss"
BUBBLE_STYLESHEET_FILENAME = "bubble_style.qss"
UI_DIR_NAME = "ui"
UI_DIR_PATH = os.path.join(APP_BASE_DIR, UI_DIR_NAME)

STYLE_PATHS_TO_CHECK = [
    os.path.join(APP_BASE_DIR, UI_DIR_NAME, STYLESHEET_FILENAME), # Check ui/style.qss
    os.path.join(APP_BASE_DIR, STYLESHEET_FILENAME) # Check root/style.qss
]
BUBBLE_STYLESHEET_PATH = os.path.join(UI_DIR_PATH, BUBBLE_STYLESHEET_FILENAME)


# --- RAG Specific Constants ---
RAG_COLLECTIONS_DIR_NAME = "rag_collections"
RAG_COLLECTIONS_PATH = os.path.join(USER_DATA_DIR, RAG_COLLECTIONS_DIR_NAME)
GLOBAL_COLLECTION_ID = "global_knowledge"
RAG_NUM_RESULTS = 5
RAG_CHUNK_SIZE = 1000
RAG_CHUNK_OVERLAP = 150
RAG_MAX_FILE_SIZE_MB = 50
MAX_SCAN_DEPTH = 5

ALLOWED_TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rst',
    '.py', '.js', '.ts', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.env',
    '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.swift', '.php', '.rb',
    '.pdf', '.docx',
}

DEFAULT_IGNORED_DIRS = {
    '.git', '.idea', '__pycache__', 'venv', 'node_modules', 'build', 'dist',
    '.pytest_cache', '.vscode', '.env', '.DS_Store', 'logs',
}
# --- End RAG Specific Constants ---

# --- Logging Constants ---
LOG_LEVEL = "DEBUG"
LOG_FILE_NAME = "ava_pys6_phase1.log"
LOG_FORMAT = '%(asctime)s.%(msecs)03d - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# --- End Logging Constants ---

# Ensure user data directory exists (critical for many operations)
try:
    os.makedirs(USER_DATA_DIR, exist_ok=True)
    logger.info(f"User data directory ensured at: {USER_DATA_DIR}")
except OSError as e:
    logger.critical(f"CRITICAL: Error creating user data directory in constants.py: {e}", exc_info=True)
    # Depending on how critical this is at import time, you might raise an error or exit.
    # For now, just logging critically.
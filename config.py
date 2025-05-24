import logging
import os
import sys
from typing import Optional

try:
    import dotenv
except ImportError:
    dotenv = None
    logging.getLogger(__name__).warning(
        "python-dotenv library not found. .env file will not be loaded. pip install python-dotenv")

logger = logging.getLogger(__name__)

if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_DOTENV_PATH = os.path.join(_BASE_DIR, '.env')


def load_config() -> dict:
    config = {}

    if dotenv and os.path.exists(_DOTENV_PATH):
        logger.info(f"Loading configuration from: {_DOTENV_PATH}")
        dotenv.load_dotenv(dotenv_path=_DOTENV_PATH)
        config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
        config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")  # For future use
        if not config.get('GEMINI_API_KEY'):
            logger.warning("GEMINI_API_KEY not found or empty in .env file.")
    else:
        if not dotenv:
            logger.info(
                ".env file support disabled (python-dotenv not installed). Checking system environment variables.")
        else:
            logger.warning(f".env file not found at {_DOTENV_PATH}. Checking system environment variables.")

        config['GEMINI_API_KEY'] = os.getenv("GEMINI_API_KEY")
        config['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")  # For future use
        if not config.get('GEMINI_API_KEY'):
            logger.warning("GEMINI_API_KEY not found in system environment variables.")

    return config


APP_CONFIG = load_config()


def get_gemini_api_key() -> Optional[str]:
    return APP_CONFIG.get("GEMINI_API_KEY")


def get_openai_api_key() -> Optional[str]:  # For future use
    return APP_CONFIG.get("OPENAI_API_KEY")


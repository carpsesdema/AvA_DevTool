import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

try:
    import qasync
except ImportError:
    # This basic print is for the earliest possible error detection.
    # A QMessageBox will be attempted later if QApplication can be initialized.
    print("[CRITICAL] qasync library not found. Please install it: pip install qasync", file=sys.stderr)
    sys.exit(1)

# --- Early Basic Logging Config (before full constants are loaded) ---
# This helps catch import errors of constants.py itself or other early issues.
# It will be reconfigured more robustly once utils.constants are loaded.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_prelim_logger = logging.getLogger(__name__)  # For early logging before full setup

try:
    from utils import constants  # utils.constants is critical
    from config import get_gemini_api_key  # To check early if API key is an issue
    from core.application_orchestrator import ApplicationOrchestrator
    from core.chat_manager import ChatManager
    from ui.main_window import MainWindow
    # ChatMessageStateHandler will be instantiated inside MainWindow or Orchestrator later
except ImportError as e:
    _prelim_logger.critical(f"Failed to import core components in main.py: {e}", exc_info=True)
    _prelim_logger.info(f"PYTHONPATH: {sys.path}")
    if 'utils.constants' not in sys.modules:  # Check if constants specifically failed
        _prelim_logger.error(
            "Failed to import 'utils.constants'. This is often a PYTHONPATH issue or "
            "you might be running 'main.py' from a subdirectory instead of the project root."
        )
    try:
        # Attempt to show a Qt message box if QApplication can be initialized
        _dummy_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Fatal Import Error",
                             f"Failed to import critical application components:\n{e}\n\n"
                             "This might be due to an incorrect execution directory or PYTHONPATH.\n"
                             "Ensure you are running from the project root directory (e.g., 'ava_pyside_project').\n\n"
                             "Check the console output and logs for more details.")
    except Exception as e_qm:
        _prelim_logger.critical(f"Failed to show import error message via QMessageBox: {e_qm}")
    sys.exit(1)  # Exit after attempting to show the message


# --- Full Logging Setup using constants ---
def setup_logging():
    """Configures the application-wide logging."""
    try:
        os.makedirs(constants.USER_DATA_DIR, exist_ok=True)
    except OSError as e_dir:
        # If this fails, log to console and continue; file logging will be disabled.
        _prelim_logger.critical(f"Could not create user data directory {constants.USER_DATA_DIR} for logging: {e_dir}")
        # No return here, let it try to set up console logging at least.

    log_file_path = os.path.join(constants.USER_DATA_DIR, constants.LOG_FILE_NAME)
    log_level_actual = getattr(logging, constants.LOG_LEVEL.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    # Set root logger to the most verbose level any handler will use
    # (e.g., if file is DEBUG and console is INFO, root should be DEBUG)
    root_logger.setLevel(
        min(log_level_actual, logging.DEBUG if constants.LOG_LEVEL.upper() == "DEBUG" else logging.INFO))

    # Clear any handlers already attached by basicConfig or previous runs in interactive sessions
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # File Handler
    try:
        file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setLevel(log_level_actual)  # File gets logs from its specified level
        file_formatter = logging.Formatter(constants.LOG_FORMAT, datefmt=constants.LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e_fh:
        _prelim_logger.error(f"Failed to set up file logger at {log_file_path}: {e_fh}. File logging disabled.")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)  # Use stdout for console
    # Make console less verbose by default, e.g., INFO or WARNING
    console_handler.setLevel(logging.INFO if constants.LOG_LEVEL.upper() in ["DEBUG", "INFO"] else logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)-8s [%(name)s] %(message)s')  # Simpler format for console
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Set levels for noisy third-party libraries after our handlers are set
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)  # Pillow can be verbose

    # Test log message after full setup, using the module-level logger
    # This logger will inherit from the root logger's configuration.
    logging.getLogger(__name__).info(
        f"Logging fully configured. Effective log level for file: {logging.getLevelName(log_level_actual)}. "
        f"Log file path: {log_file_path}"
    )


setup_logging()
# Now, get the logger for this module, which will use the handlers configured above.
logger = logging.getLogger(__name__)


async def async_main():
    logger.info(f"--- Starting {constants.APP_NAME} v{constants.APP_VERSION} (PySide6 - Phase 1) ---")

    # API Key Check (early warning, does not prevent app launch for P1)
    if not get_gemini_api_key():
        logger.warning(
            "Gemini API Key is NOT configured. Please set GEMINI_API_KEY in .env file "
            "or as an environment variable. Gemini-based features will not work."
        )
        # For Phase 1, we might allow the app to run but show a persistent UI warning.
        # For now, just a log warning. A QMessageBox could be added here.

    app = QApplication.instance()
    if app is None:
        # Standard Qt attributes for High DPI support
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    app.setApplicationName(constants.APP_NAME)
    app.setApplicationVersion(constants.APP_VERSION)

    try:
        app_icon_path = os.path.join(constants.ASSETS_PATH, "ava_logo.svg")  # Assuming you have a logo
        if os.path.exists(app_icon_path):
            app.setWindowIcon(QIcon(app_icon_path))
        else:
            logger.warning(f"Application icon not found at {app_icon_path}. Using default system icon.")
    except Exception as e_icon:
        logger.error(f"Error setting application icon: {e_icon}", exc_info=True)

    # Declare components to be initialized
    main_window: Optional[MainWindow] = None
    chat_manager: Optional[ChatManager] = None
    app_orchestrator: Optional[ApplicationOrchestrator] = None

    try:
        logger.info("Instantiating ApplicationOrchestrator...")

        # For Phase 1, SessionService and UploadService are not fully used but orchestrator expects them.
        # Define basic placeholder versions for P1 if full services are not yet implemented.

        # Basic SessionService for P1 (can be fleshed out later in services/session_service.py)
        class P1SessionService:  # Minimal implementation for P1
            def get_last_session(self): return None, None, None, None  # model, pers, proj_data, extra_data

            def save_last_session(self, *args, **kwargs): pass

            def __init__(self): logger.debug("P1SessionService (placeholder) initialized.")

        # Basic UploadService for P1
        class P1UploadService:  # Minimal implementation for P1
            def is_vector_db_ready(self, *args, **kwargs): return False

            def __init__(self): logger.debug("P1UploadService (placeholder) initialized.")

        session_service_p1 = P1SessionService()
        upload_service_p1 = P1UploadService()

        # FIXED: Use positional arguments instead of keyword arguments
        app_orchestrator = ApplicationOrchestrator(
            session_service_p1,  # First positional argument
            upload_service_p1    # Second positional argument
        )
        logger.info("ApplicationOrchestrator instantiated.")

        logger.info("Instantiating ChatManager...")
        chat_manager = ChatManager(orchestrator=app_orchestrator)
        logger.info("ChatManager instantiated.")

        logger.info("Instantiating MainWindow...")
        main_window = MainWindow(chat_manager=chat_manager, app_base_path=constants.APP_BASE_DIR)
        logger.info("MainWindow instantiated.")

    except Exception as e_init:
        logger.critical(" ***** FATAL ERROR DURING CORE COMPONENT INSTANTIATION ***** ", exc_info=True)
        QMessageBox.critical(None, "Fatal Initialization Error",
                             f"Failed during application component setup:\n{e_init}\n\n"
                             f"Please check logs at {os.path.join(constants.USER_DATA_DIR, constants.LOG_FILE_NAME)}")
        if app: app.quit()  # Ensure app quits if critical components fail
        return 1  # Indicate error

    if chat_manager:
        # Perform late initialization for ChatManager (e.g., loading last settings, configuring backends)
        # This is often done after the main window is shown or via a short QTimer delay
        # to ensure the UI event loop is running.
        QTimer.singleShot(100, chat_manager.initialize)
        logger.info("Scheduled ChatManager late initialization.")
    else:
        logger.critical("ChatManager failed to instantiate. Application cannot continue.")
        if app: app.quit()
        return 1

    if main_window:
        main_window.show()  # Show the main window
        logger.info("--- Main Window Shown ---")
    else:
        logger.critical("MainWindow instance not created, cannot show window.")
        if app: app.quit()
        return 1

    logger.info("--- async_main: Entering main blocking phase (await asyncio.Future()) ---")
    await asyncio.Future()  # This keeps the qasync event loop running until the app is closed.

    logger.info("--- async_main returning, Application Event Loop should be finishing ---")
    return 0  # Indicate success


if __name__ == "__main__":
    # This initial log might go to console before file handler is fully up if there's an early issue,
    # but subsequent logs from within async_main and components will use the configured handlers.
    logger.info(f"Application starting (__name__ == '__main__').")

    q_app_instance = QApplication.instance()  # Check if an instance already exists
    if q_app_instance is None:
        q_app_instance = QApplication(sys.argv)  # Create one if not

    event_loop: Optional[qasync.QEventLoop] = None
    exit_code = 1  # Default to error exit code

    try:
        # Recommended for Windows asyncio integration with Qt
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            logger.debug("Setting WindowsSelectorEventLoopPolicy for asyncio compatibility.")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        event_loop = qasync.QEventLoop(q_app_instance)
        asyncio.set_event_loop(event_loop)
        logger.info("qasync event loop configured and set as the current asyncio event loop.")

        with event_loop:  # Context manager ensures loop is properly managed
            logger.info("Running qasync event loop until async_main completes...")
            exit_code = event_loop.run_until_complete(async_main())
            logger.info(f"async_main completed. Exit code from run_until_complete: {exit_code}")

    except RuntimeError as e_rt_loop:
        # Handle cases where the event loop might already be running (e.g., in some IDEs)
        if "cannot be nested" in str(e_rt_loop).lower() or "already running" in str(e_rt_loop).lower():
            logger.warning(
                f"qasync event loop issue: {e_rt_loop}. The loop may already be running (e.g., in an interactive environment).")
            # If an application window is already active, this might not be a fatal error in certain contexts.
            if QApplication.instance() and QApplication.instance().activeWindow():
                logger.info("Application window seems to be active despite event loop warning.")
                # For a standalone app, this usually indicates a problem, so still set error exit_code.
                exit_code = 1
            else:
                exit_code = 1  # Treat as error
        else:
            # Other RuntimeErrors are more critical
            logger.critical(f"RuntimeError during qasync event loop execution: {e_rt_loop}", exc_info=True)
            exit_code = 1
    except Exception as e_outer_loop:
        logger.critical(f"Unhandled exception during application startup or event loop run: {e_outer_loop}",
                        exc_info=True)
        # Attempt to show a message box for critical errors if possible
        try:
            QMessageBox.critical(None, "Critical Application Error",
                                 f"A critical error occurred:\n{e_outer_loop}\n\n"
                                 "The application will now exit. Please check the logs.")
        except Exception:
            pass  # If even QMessageBox fails, console log is the best we can do
        exit_code = 1
    finally:
        logger.info(f"Application attempting to exit with code: {exit_code}")
        if event_loop and event_loop.is_running():
            logger.info("Event loop is still running in the 'finally' block; attempting to close it.")
            event_loop.close()

        # Ensure all log handlers are flushed and closed properly
        logging.shutdown()
        sys.exit(exit_code)
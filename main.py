# main.py
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# --- Ensure the project root is in sys.path for absolute imports ---
# This helps resolve imports like 'from services.X import Y'
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End project root sys.path setup ---

# --- PySide6 Imports (Explicitly added back for clarity and robustness) ---
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox
# --- End PySide6 Imports ---

try:
    import qasync
except ImportError:
    print("[CRITICAL] qasync library not found. Please install it: pip install qasync", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - [%(name)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
_prelim_logger = logging.getLogger(__name__)

try:
    from utils import constants
    from config import get_gemini_api_key
    from core.application_orchestrator import ApplicationOrchestrator
    from core.chat_manager import ChatManager
    from ui.main_window import MainWindow
    from services.project_service import ProjectManager
except ImportError as e:
    _prelim_logger.critical(f"Failed to import core components in main.py: {e}", exc_info=True)
    _prelim_logger.info(f"PYTHONPATH: {sys.path}")
    if 'utils.constants' not in sys.modules:
        _prelim_logger.error(
            "Failed to import 'utils.constants'. This is often a PYTHONPATH issue or "
            "you might be running 'main.py' from a subdirectory instead of the project root."
        )
    # The QMessageBox here relies on QApplication, which is now imported explicitly above.
    try:
        _dummy_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Fatal Import Error",
                             f"Failed to import critical application components:\n{e}\n\n"
                             "This might be due to an incorrect execution directory or PYTHONPATH.\n"
                             "Ensure you are running from the project root directory.\n\n"
                             "Check the console output and logs for more details.")
    except Exception as e_qm:
        _prelim_logger.critical(f"Failed to show import error message via QMessageBox: {e_qm}")
    sys.exit(1)


def setup_logging():
    try:
        os.makedirs(constants.USER_DATA_DIR, exist_ok=True)
    except OSError as e_dir:
        _prelim_logger.critical(f"Could not create user data directory {constants.USER_DATA_DIR} for logging: {e_dir}")

    log_file_path = os.path.join(constants.USER_DATA_DIR, constants.LOG_FILE_NAME)
    log_level_actual = getattr(logging, constants.LOG_LEVEL.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(
        min(log_level_actual, logging.DEBUG if constants.LOG_LEVEL.upper() == "DEBUG" else logging.INFO))

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    try:
        file_handler = RotatingFileHandler(log_file_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        file_handler.setLevel(log_level_actual)
        file_formatter = logging.Formatter(constants.LOG_FORMAT, datefmt=constants.LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e_fh:
        _prelim_logger.error(f"Failed to set up file logger at {log_file_path}: {e_fh}. File logging disabled.")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if constants.LOG_LEVEL.upper() in ["DEBUG", "INFO"] else logging.WARNING)
    console_formatter = logging.Formatter('%(levelname)-8s [%(name)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.INFO)

    logging.getLogger(__name__).info(
        f"Logging fully configured. Effective log level for file: {logging.getLevelName(log_level_actual)}. "
        f"Log file path: {log_file_path}"
    )


setup_logging()
logger = logging.getLogger(__name__)


async def async_main():
    logger.info(f"--- Starting {constants.APP_NAME} v{constants.APP_VERSION} ---")

    if not get_gemini_api_key():
        logger.warning(
            "Gemini API Key is NOT configured. Please set GEMINI_API_KEY in .env file "
            "or as an environment variable. Gemini-based features will not work."
        )

    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    app.setApplicationName(constants.APP_NAME)
    app.setApplicationVersion(constants.APP_VERSION)

    try:
        app_icon_path = os.path.join(constants.ASSETS_PATH, "ava_logo.svg")
        if os.path.exists(app_icon_path):
            app.setWindowIcon(QIcon(app_icon_path))
        else:
            logger.warning(f"Application icon not found at {app_icon_path}. Using default system icon.")
    except Exception as e_icon:
        logger.error(f"Error setting application icon: {e_icon}", exc_info=True)

    main_window: Optional[MainWindow] = None
    chat_manager: Optional[ChatManager] = None
    app_orchestrator: Optional[ApplicationOrchestrator] = None
    project_manager: Optional[ProjectManager] = None

    try:
        logger.info("Instantiating ProjectManager...")
        project_manager = ProjectManager()
        logger.info("ProjectManager instantiated.")

        logger.info("Instantiating ApplicationOrchestrator...")

        # Removed P1UploadService placeholder as ApplicationOrchestrator now initializes real UploadService
        app_orchestrator = ApplicationOrchestrator(
            project_manager=project_manager
        )
        logger.info("ApplicationOrchestrator instantiated.")

        logger.info("Instantiating ChatManager...")
        chat_manager = ChatManager(orchestrator=app_orchestrator)
        logger.info("ChatManager instantiated.")

        if app_orchestrator:
            app_orchestrator.set_chat_manager(chat_manager)

        logger.info("Instantiating MainWindow...")
        main_window = MainWindow(chat_manager=chat_manager, app_base_path=constants.APP_BASE_DIR)
        logger.info("MainWindow instantiated.")

        if app_orchestrator:
            app_orchestrator.initialize_application_state()


    except Exception as e_init:
        logger.critical(" ***** FATAL ERROR DURING CORE COMPONENT INSTANTIATION ***** ", exc_info=True)
        QMessageBox.critical(None, "Fatal Initialization Error",
                             f"Failed during application component setup:\n{e_init}\n\n"
                             f"Please check logs at {os.path.join(constants.USER_DATA_DIR, constants.LOG_FILE_NAME)}")
        if app: app.quit()
        return 1

    if chat_manager:
        # ChatManager's own late init for backend config
        QTimer.singleShot(50, chat_manager.initialize)
        logger.info("Scheduled ChatManager late initialization for backends.")
    else:
        logger.critical("ChatManager failed to instantiate. Application cannot continue.")
        if app: app.quit()
        return 1

    # Orchestrator's app state init (project/session) is now called above,
    # but UI might need a slight delay to fully show before status updates hit.
    # The `showEvent` in MainWindow handles its own post-show QTimer calls.

    if main_window:
        main_window.show()
        logger.info("--- Main Window Shown ---")
    else:
        logger.critical("MainWindow instance not created, cannot show window.")
        if app: app.quit()
        return 1

    logger.info("--- async_main: Entering main blocking phase (await asyncio.Future()) ---")
    await asyncio.Future()

    logger.info("--- async_main returning, Application Event Loop should be finishing ---")
    return 0


if __name__ == "__main__":
    logger.info(f"Application starting (__name__ == '__main__').")

    q_app_instance = QApplication.instance()
    if q_app_instance is None:
        q_app_instance = QApplication(sys.argv)

    event_loop: Optional[qasync.QEventLoop] = None
    exit_code = 1

    try:
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            logger.debug("Setting WindowsSelectorEventLoopPolicy for asyncio compatibility.")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        event_loop = qasync.QEventLoop(q_app_instance)
        asyncio.set_event_loop(event_loop)
        logger.info("qasync event loop configured and set as the current asyncio event loop.")

        with event_loop:
            logger.info("Running qasync event loop until async_main completes...")
            exit_code = event_loop.run_until_complete(async_main())
            logger.info(f"async_main completed. Exit code from run_until_complete: {exit_code}")

    except RuntimeError as e_rt_loop:
        if "cannot be nested" in str(e_rt_loop).lower() or "already running" in str(e_rt_loop).lower():
            logger.warning(
                f"qasync event loop issue: {e_rt_loop}. The loop may already be running.")
            if QApplication.instance() and QApplication.instance().activeWindow():  # type: ignore
                logger.info("Application window seems to be active despite event loop warning.")
                exit_code = 1  # Still an issue for standalone app
            else:
                exit_code = 1
        else:
            logger.critical(f"RuntimeError during qasync event loop execution: {e_rt_loop}", exc_info=True)
            exit_code = 1
    except Exception as e_outer_loop:
        logger.critical(f"Unhandled exception during application startup or event loop run: {e_outer_loop}",
                        exc_info=True)
        try:
            QMessageBox.critical(None, "Critical Application Error",
                                 f"A critical error occurred:\n{e_outer_loop}\n\n"
                                 "The application will now exit. Please check the logs.")
        except Exception:
            pass
        exit_code = 1
    finally:
        logger.info(f"Application attempting to exit with code: {exit_code}")
        if event_loop and event_loop.is_running():
            logger.info("Event loop is still running in the 'finally' block; attempting to close it.")
            event_loop.close()

        logging.shutdown()
        sys.exit(exit_code)
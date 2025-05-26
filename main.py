# main.py - Fixed for async initialization and better error handling
import asyncio
import logging
import os
import signal
import sys
import traceback
from pathlib import Path


# Set up logging before any other imports
def setup_logging():
    """Set up logging configuration"""
    log_level = logging.INFO
    if os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes'):
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)-8s - [%(name)s:%(module)s.%(funcName)s:%(lineno)d] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

    # Reduce noise from some third-party libraries
    logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)

# Now import Qt and other modules
try:
    import qasync
    from PySide6.QtWidgets import QApplication, QMessageBox
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QIcon
except ImportError as e:
    logger.critical(f"Failed to import required Qt/qasync modules: {e}")
    sys.exit(1)

try:
    from core.application_orchestrator import ApplicationOrchestrator
    from core.chat_manager import ChatManager
    from services.project_service import ProjectManager
    from ui.main_window import MainWindow
    from utils import constants
except ImportError as e:
    logger.critical(f"Failed to import application modules: {e}")
    traceback.print_exc()
    sys.exit(1)


class AvAApplication:
    """Main application class that manages initialization and cleanup"""

    def __init__(self):
        self.app = None
        self.event_loop = None
        self.main_window = None
        self.orchestrator = None
        self.chat_manager = None
        self.project_manager = None
        self._initialization_complete = False
        self._shutdown_requested = False

    async def initialize_async(self):
        """Initialize the application asynchronously"""
        try:
            logger.info("Starting async application initialization...")

            # Create Qt application
            if QApplication.instance() is None:
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()

            self.app.setApplicationName(constants.APP_NAME)
            self.app.setApplicationVersion(constants.APP_VERSION)
            self.app.setOrganizationName("AvA DevTool")

            # Set application icon if available
            try:
                icon_path = os.path.join(constants.ASSETS_PATH, "Synchat.ico")
                if os.path.exists(icon_path):
                    self.app.setWindowIcon(QIcon(icon_path))
            except Exception as e:
                logger.warning(f"Could not set application icon: {e}")

            # Create qasync event loop
            self.event_loop = qasync.QEventLoop(self.app)
            asyncio.set_event_loop(self.event_loop)

            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()

            # Initialize core services
            await self._initialize_core_services()

            # Initialize UI
            await self._initialize_ui()

            # Final initialization steps
            await self._finalize_initialization()

            self._initialization_complete = True
            logger.info("Application initialization completed successfully")

        except Exception as e:
            logger.critical(f"Failed to initialize application: {e}")
            traceback.print_exc()
            raise

    async def _initialize_core_services(self):
        """Initialize core services with proper async handling"""
        logger.info("Initializing core services...")

        # Create project manager first
        self.project_manager = ProjectManager()

        # Create application orchestrator
        self.orchestrator = ApplicationOrchestrator(project_manager=self.project_manager)

        # Create chat manager and link it to orchestrator
        self.chat_manager = ChatManager(self.orchestrator)
        self.orchestrator.set_chat_manager(self.chat_manager)

        # Wait a bit for any async initialization in services
        await asyncio.sleep(0.1)

        logger.info("Core services initialized")

    async def _initialize_ui(self):
        """Initialize the UI components"""
        logger.info("Initializing UI...")

        # Create main window
        self.main_window = MainWindow(
            chat_manager=self.chat_manager,
            app_base_path=os.getcwd()
        )

        # Show the main window
        self.main_window.show()

        # Wait for UI to stabilize
        await asyncio.sleep(0.1)

        logger.info("UI initialized")

    async def _finalize_initialization(self):
        """Finalize initialization steps"""
        logger.info("Finalizing initialization...")

        # Initialize chat manager (configure backends, etc.)
        self.chat_manager.initialize()

        # Initialize application state (projects, sessions)
        self.orchestrator.initialize_application_state()

        # Wait for any remaining async operations
        await asyncio.sleep(0.2)

        logger.info("Initialization finalized")

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        if sys.platform != 'win32':
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._signal_handler)
        else:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self._shutdown_requested = True
        if self.event_loop and self.event_loop.is_running():
            # Schedule shutdown on the event loop
            asyncio.create_task(self.shutdown_async())

    async def shutdown_async(self):
        """Perform async shutdown"""
        if self._shutdown_requested:
            return  # Already shutting down

        self._shutdown_requested = True
        logger.info("Starting async shutdown...")

        try:
            # Close main window
            if self.main_window:
                self.main_window.close()

            # Give time for cleanup
            await asyncio.sleep(0.1)

            # Quit application
            if self.app:
                self.app.quit()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("Async shutdown completed")

    async def run_async(self):
        """Run the application event loop"""
        try:
            logger.info("Starting application event loop...")

            # Create a task to monitor for shutdown
            shutdown_task = asyncio.create_task(self._monitor_shutdown())

            # Run until shutdown is requested
            while not self._shutdown_requested and self.app and not self.app.closingDown():
                await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting
                self.app.processEvents()

            # Cancel shutdown monitor
            shutdown_task.cancel()

            logger.info("Event loop completed")
            return 0

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            return 0
        except Exception as e:
            logger.critical(f"Error in event loop: {e}")
            traceback.print_exc()
            return 1

    async def _monitor_shutdown(self):
        """Monitor for shutdown conditions"""
        try:
            while not self._shutdown_requested:
                if self.app and self.app.closingDown():
                    self._shutdown_requested = True
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass  # Normal cancellation


async def async_main():
    """Main async entry point"""
    app_instance = AvAApplication()

    try:
        # Initialize the application
        await app_instance.initialize_async()

        # Run the application
        exit_code = await app_instance.run_async()

        # Shutdown
        await app_instance.shutdown_async()

        return exit_code

    except Exception as e:
        logger.critical(f"Unhandled exception in async_main: {e}")
        traceback.print_exc()

        # Try to show error dialog if possible
        try:
            if app_instance.app:
                QMessageBox.critical(
                    None,
                    "Critical Error",
                    f"Application failed to start:\n{str(e)}\n\nCheck logs for details."
                )
        except:
            pass  # If we can't show the dialog, just continue

        return 1


def main():
    """Main entry point"""
    exit_code = 1

    try:
        logger.info(f"Starting {constants.APP_NAME} v{constants.APP_VERSION}")

        # Run the async main function
        if sys.platform == 'win32':
            # On Windows, use the WindowsProactorEventLoopPolicy for better compatibility
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # Create a new event loop and run the async main
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            exit_code = loop.run_until_complete(async_main())
        finally:
            # Clean up the event loop
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            except Exception as cleanup_error:
                logger.warning(f"Error during loop cleanup: {cleanup_error}")
            finally:
                loop.close()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        exit_code = 0
    except Exception as e:
        logger.critical(f"Unhandled exception during application startup or event loop run: {e}")
        traceback.print_exc()
        exit_code = 1

    logger.info(f"Application attempting to exit with code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
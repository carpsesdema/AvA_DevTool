# ui/loading_overlay.py
import logging
import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QMovie, QFont
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout

try:
    from utils import constants
except ImportError:
    class constants_fallback:
        ASSETS_PATH = "assets"
        CHAT_FONT_FAMILY = "Segoe UI"
        CHAT_FONT_SIZE = 10


    constants = constants_fallback
    logging.getLogger(__name__).warning("LoadingOverlay: Could not import constants, using fallback values.")

logger = logging.getLogger(__name__)


class LoadingOverlay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("LoadingOverlay")

        # Set up the overlay properties
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Initialize components
        self._movie: Optional[QMovie] = None
        self._gif_label: Optional[QLabel] = None
        self._message_label: Optional[QLabel] = None
        self._animation_timer: Optional[QTimer] = None

        self._init_ui()
        self._load_animation()

        # Start hidden
        self.hide()

        logger.info("LoadingOverlay initialized")

    def _init_ui(self):
        """Initialize the UI components"""
        # Set up the overlay background
        self.setStyleSheet("""
            QWidget#LoadingOverlay {
                background-color: rgba(13, 17, 23, 180);  /* Semi-transparent dark */
                border: none;
            }

            QLabel#LoadingGif {
                background-color: transparent;
                border: none;
            }

            QLabel#LoadingMessage {
                background-color: transparent;
                color: #c9d1d9;
                font-weight: bold;
                border: none;
                padding: 10px;
            }
        """)

        # Main layout - center everything
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setContentsMargins(50, 50, 50, 50)

        # Container for the loading content
        content_widget = QWidget()
        content_widget.setObjectName("LoadingContent")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.setSpacing(20)

        # Animated GIF label
        self._gif_label = QLabel()
        self._gif_label.setObjectName("LoadingGif")
        self._gif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_label.setFixedSize(60, 60)  # Size for the gif

        # Message label
        self._message_label = QLabel("Loading...")
        self._message_label.setObjectName("LoadingMessage")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Set font for message
        font = QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE + 2)
        font.setBold(True)
        self._message_label.setFont(font)

        # Add to layout
        content_layout.addWidget(self._gif_label)
        content_layout.addWidget(self._message_label)

        main_layout.addWidget(content_widget)

        self.setLayout(main_layout)

    def _load_animation(self):
        """Load the animated GIF"""
        gif_path = os.path.join(constants.ASSETS_PATH, "twisting_orb_loading.gif")

        if not os.path.exists(gif_path):
            logger.error(f"Loading GIF not found at: {gif_path}")
            # Fallback to text-based loading indicator
            self._gif_label.setText("🔄")
            self._gif_label.setStyleSheet("font-size: 24px;")
            return

        try:
            self._movie = QMovie(gif_path)

            if not self._movie.isValid():
                logger.error(f"Invalid GIF file: {gif_path}")
                self._gif_label.setText("⚙️")
                self._gif_label.setStyleSheet("font-size: 24px;")
                return

            # Set up the movie
            self._movie.setScaledSize(self._gif_label.size())
            self._gif_label.setMovie(self._movie)

            logger.info(f"Loading animation loaded successfully from: {gif_path}")

        except Exception as e:
            logger.error(f"Error loading animation: {e}")
            # Fallback to emoji
            self._gif_label.setText("⚡")
            self._gif_label.setStyleSheet("font-size: 24px;")

    def show_loading(self, message: str = "Loading..."):
        """Show the loading overlay with a message"""
        if self._message_label:
            self._message_label.setText(message)

        # Resize to cover parent
        if self.parent():
            self.resize(self.parent().size())

        # Start animation if available
        if self._movie and self._movie.isValid():
            self._movie.start()
            logger.debug("Started loading animation")

        # Show the overlay
        self.show()
        self.raise_()  # Bring to front

        logger.info(f"Loading overlay shown with message: {message}")

    def hide_loading(self):
        """Hide the loading overlay"""
        # Stop animation
        if self._movie and self._movie.isValid():
            self._movie.stop()
            logger.debug("Stopped loading animation")

        # Hide the overlay
        self.hide()
        logger.info("Loading overlay hidden")

    def update_message(self, message: str):
        """Update the loading message without hiding/showing"""
        if self._message_label:
            self._message_label.setText(message)
            logger.debug(f"Updated loading message: {message}")

    def resizeEvent(self, event):
        """Handle resize events to stay covering the parent"""
        super().resizeEvent(event)

        # If we have a parent, resize to match it
        if self.parent() and self.isVisible():
            self.resize(self.parent().size())

    def showEvent(self, event):
        """Handle show event"""
        super().showEvent(event)

        # Ensure we cover the parent window
        if self.parent():
            self.resize(self.parent().size())
            self.move(0, 0)  # Position at top-left of parent
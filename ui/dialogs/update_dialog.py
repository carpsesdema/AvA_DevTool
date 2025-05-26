# ui/dialogs/update_dialog.py
import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QProgressBar, QFrame, QMessageBox, QWidget
)

try:
    from utils import constants
    from services.update_service import UpdateInfo
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in UpdateDialog: {e}", exc_info=True)


    # Fallback for constants if needed
    class constants_fallback:
        CHAT_FONT_FAMILY = "Arial"
        CHAT_FONT_SIZE = 10
        APP_NAME = "AvA"
        APP_VERSION = "0.1.0"


    constants = constants_fallback
    raise

logger = logging.getLogger(__name__)


class UpdateDialog(QDialog):
    """Dialog for showing available updates and handling download/installation"""

    download_requested = Signal(object)  # UpdateInfo
    install_requested = Signal(str)  # file_path
    restart_requested = Signal()

    def __init__(self, update_info: UpdateInfo, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.update_info = update_info
        self.downloaded_file_path: Optional[str] = None

        self.setWindowTitle(f"Update Available - {constants.APP_NAME}")
        self.setObjectName("UpdateDialog")
        self.setMinimumSize(500, 400)
        self.setMaximumSize(600, 500)
        self.setModal(True)

        self._init_ui()
        self._connect_signals()

        # Auto-scroll changelog to top
        QTimer.singleShot(100, lambda: self.changelog_text.verticalScrollBar().setValue(0))

        logger.info(f"UpdateDialog initialized for version {update_info.version}")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header section
        header_layout = QHBoxLayout()

        # Icon (you can add an update icon here)
        icon_label = QLabel("🚀")
        icon_label.setFont(QFont(constants.CHAT_FONT_FAMILY, 32))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFixedSize(60, 60)
        header_layout.addWidget(icon_label)

        # Title and version info
        title_layout = QVBoxLayout()
        title_label = QLabel("Update Available!")
        title_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE + 4, QFont.Weight.Bold))
        title_layout.addWidget(title_label)

        version_label = QLabel(f"Version {self.update_info.version} is now available")
        version_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE + 1))
        version_label.setStyleSheet("color: #61AFEF;")
        title_layout.addWidget(version_label)

        current_label = QLabel(f"Current version: {constants.APP_VERSION}")
        current_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        current_label.setStyleSheet("color: #888888;")
        title_layout.addWidget(current_label)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("color: #30363d;")
        layout.addWidget(separator)

        # Update details
        details_layout = QVBoxLayout()

        size_label = QLabel(f"Download size: {self.update_info.file_size_mb:.1f} MB")
        size_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        details_layout.addWidget(size_label)

        if self.update_info.critical:
            critical_label = QLabel("⚠️ This is a critical security update")
            critical_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE, QFont.Weight.Bold))
            critical_label.setStyleSheet("color: #e06c75;")
            details_layout.addWidget(critical_label)

        layout.addLayout(details_layout)

        # Changelog section
        changelog_label = QLabel("What's New:")
        changelog_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE, QFont.Weight.Bold))
        layout.addWidget(changelog_label)

        self.changelog_text = QTextEdit()
        self.changelog_text.setObjectName("UpdateChangelogText")
        self.changelog_text.setReadOnly(True)
        self.changelog_text.setMaximumHeight(150)
        self.changelog_text.setPlainText(self.update_info.changelog or "No changelog provided.")
        self.changelog_text.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        layout.addWidget(self.changelog_text)

        # Progress section (initially hidden)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout(self.progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)

        self.status_label = QLabel("Ready to download")
        self.status_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        progress_layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        layout.addWidget(self.progress_widget)

        # Button section
        button_layout = QHBoxLayout()

        self.later_button = QPushButton("Remind Me Later")
        self.later_button.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
        button_layout.addWidget(self.later_button)

        self.skip_button = QPushButton("Skip This Version")
        self.skip_button.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
        button_layout.addWidget(self.skip_button)

        button_layout.addStretch()

        self.download_button = QPushButton("Download Update")
        self.download_button.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
        self.download_button.setStyleSheet("""
            QPushButton {
                background-color: #00e676;
                color: #0d1117;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #26ff87;
            }
            QPushButton:pressed {
                background-color: #00b359;
            }
            QPushButton:disabled {
                background-color: #21262d;
                color: #6e7681;
            }
        """)
        button_layout.addWidget(self.download_button)

        self.install_button = QPushButton("Install & Restart")
        self.install_button.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
        self.install_button.setVisible(False)
        self.install_button.setStyleSheet("""
            QPushButton {
                background-color: #61afef;
                color: #0d1117;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #7cc7ff;
            }
            QPushButton:pressed {
                background-color: #4a9eff;
            }
        """)
        button_layout.addWidget(self.install_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _connect_signals(self):
        self.later_button.clicked.connect(self.reject)
        self.skip_button.clicked.connect(self._skip_version)
        self.download_button.clicked.connect(self._start_download)
        self.install_button.clicked.connect(self._install_update)

    @Slot()
    def _start_download(self):
        logger.info("User clicked download update")
        self.download_button.setEnabled(False)
        self.download_button.setText("Downloading...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting download...")

        self.download_requested.emit(self.update_info)

    @Slot()
    def _install_update(self):
        if not self.downloaded_file_path:
            logger.error("No downloaded file to install")
            return

        reply = QMessageBox.question(
            self, "Install Update",
            "The application will restart to complete the update. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            logger.info("User confirmed update installation")
            self.install_requested.emit(self.downloaded_file_path)

    @Slot()
    def _skip_version(self):
        reply = QMessageBox.question(
            self, "Skip Version",
            f"Are you sure you want to skip version {self.update_info.version}?\n"
            "You won't be notified about this version again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # TODO: Save skipped version to settings
            logger.info(f"User skipped version {self.update_info.version}")
            self.reject()

    @Slot(int)
    def update_progress(self, percentage: int):
        self.progress_bar.setValue(percentage)
        if percentage < 100:
            self.status_label.setText(f"Downloading... {percentage}%")

    @Slot(str)
    def update_status(self, message: str):
        self.status_label.setText(message)

    @Slot(str)
    def download_completed(self, file_path: str):
        self.downloaded_file_path = file_path
        self.progress_bar.setValue(100)
        self.status_label.setText("Download completed!")

        # Hide download button, show install button
        self.download_button.setVisible(False)
        self.install_button.setVisible(True)

        # Update later button text
        self.later_button.setText("Install Later")

    @Slot(str)
    def download_failed(self, error_message: str):
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Download failed: {error_message}")
        self.download_button.setEnabled(True)
        self.download_button.setText("Retry Download")

    def closeEvent(self, event):
        # If download is in progress, ask for confirmation
        if self.download_button.isEnabled() == False and not self.downloaded_file_path:
            reply = QMessageBox.question(
                self, "Cancel Download",
                "Download is in progress. Cancel it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        super().closeEvent(event)
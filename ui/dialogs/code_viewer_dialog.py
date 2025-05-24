# ui/dialogs/code_viewer_dialog.py
import logging
from datetime import datetime
from typing import Optional, Dict

from PySide6.QtCore import Qt, QTimer, Slot, Signal
from PySide6.QtGui import QIcon, QFontDatabase, QTextOption, QFont
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QSplitter,
    QApplication, QMessageBox, QTreeWidget, QTreeWidgetItem, QWidget, QLabel
)

try:
    from utils import constants
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE
except ImportError as e:
    logging.error(f"Error importing dependencies in code_viewer_dialog.py: {e}. Check relative paths.")


    class constants:
        CHAT_FONT_FAMILY = "Arial"
        CHAT_FONT_SIZE = 10

# Simple fallback icons since we don't have the widgets module
COPY_ICON = QIcon()
CHECK_ICON = QIcon()
APPLY_ICON = QIcon()

logger = logging.getLogger(__name__)


class CodeViewerWindow(QDialog):
    CODE_CONTENT_ROLE = Qt.ItemDataRole.UserRole + 10
    # Signal for applying changes
    # Emits: project_id, relative_filepath, new_content, focus_prefix
    apply_change_requested = Signal(str, str, str, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Generated Code Viewer")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(800, 600)
        self.setModal(False)

        self._file_contents: Dict[str, str] = {}  # filename: content
        self._original_file_contents: Dict[str, Optional[str]] = {}  # filename: original_content or None if new
        self._current_filename: Optional[str] = None
        self._current_project_id_for_apply: Optional[str] = None
        self._current_focus_prefix_for_apply: Optional[str] = None
        self._current_content_is_modification: bool = False  # Is the current view an AI mod?

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        code_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        code_font.setPointSize(constants.CHAT_FONT_SIZE)
        self.code_font = code_font

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.file_tree = QTreeWidget()
        self.file_tree.setObjectName("CodeFileTree")
        self.file_tree.setHeaderLabels(["Generated Files"])
        self.file_tree.setMinimumWidth(250)
        self.file_tree.header().setStretchLastSection(True)

        self.code_edit = QTextEdit()
        self.code_edit.setObjectName("CodeViewerEdit")
        self.code_edit.setReadOnly(True)
        self.code_edit.setFont(self.code_font)
        self.code_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.code_edit.setWordWrapMode(QTextOption.WrapMode.NoWrap)

        self.copy_button = QPushButton("📋 Copy Code")
        self.copy_button.setToolTip("Copy the code currently shown")
        self.copy_button.setEnabled(False)

        self.apply_button = QPushButton("💾 Apply & Save")
        self.apply_button.setToolTip("Save this AI-generated file to the project")
        self.apply_button.setEnabled(False)
        self.apply_button.setObjectName("ApplyChangeButton")

        self.clear_button = QPushButton("🗑️ Clear All")
        self.clear_button.setToolTip("Remove all listed files")
        self.clear_button.setEnabled(False)

        self.close_button = QPushButton("❌ Close")
        self.close_button.setToolTip("Hide this window")

    def _init_layout(self):
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("🤖 AI Generated Code Files")
        header_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE + 1))
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Main content
        self.splitter.addWidget(self.file_tree)
        self.splitter.addWidget(self.code_edit)
        self.splitter.setSizes([280, 520])
        layout.addWidget(self.splitter, 1)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.apply_button)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _connect_signals(self):
        self.file_tree.currentItemChanged.connect(self._display_selected_file_content)
        self.copy_button.clicked.connect(self._copy_selected_code_with_feedback)
        self.apply_button.clicked.connect(self._handle_apply_change)
        self.clear_button.clicked.connect(self.clear_viewer)
        self.close_button.clicked.connect(self.hide)

    def update_or_add_file(self,
                           filename: str,
                           content: str,
                           is_ai_modification: bool = False,
                           original_content: Optional[str] = None,
                           project_id_for_apply: Optional[str] = None,
                           focus_prefix_for_apply: Optional[str] = None):
        """Main method called by MainWindow when PlanAndCodeCoordinator generates a file"""
        if not filename:
            logger.warning("Attempted to add/update file with empty filename.")
            return

        logger.info(f"CodeViewer: Adding/updating file '{filename}' (AI modification: {is_ai_modification})")

        self._file_contents[filename] = content
        if is_ai_modification:
            self._original_file_contents[filename] = original_content

        self.clear_button.setEnabled(True)

        # Find existing item or create new one
        found_item = None
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            if item and item.text(0) == filename:
                found_item = item
                break

        if found_item:
            logger.debug(f"Updating existing file in Code Viewer: {filename}")
            self.file_tree.setCurrentItem(found_item)
        else:
            logger.debug(f"Adding new file to Code Viewer: {filename}")
            new_item = QTreeWidgetItem(self.file_tree)
            new_item.setText(0, filename)
            new_item.setToolTip(0, f"Generated: {filename}")

            # Add a visual indicator for AI-generated files
            if is_ai_modification:
                new_item.setText(0, f"🤖 {filename}")

            self.file_tree.setCurrentItem(new_item)

        # Store context for apply functionality
        if self.file_tree.currentItem() and self.file_tree.currentItem().text(0).endswith(filename):
            self._current_content_is_modification = is_ai_modification
            self._current_project_id_for_apply = project_id_for_apply if is_ai_modification else None
            self._current_focus_prefix_for_apply = focus_prefix_for_apply if is_ai_modification else None
            self.apply_button.setEnabled(is_ai_modification and bool(content))

        # Show the window if it's not visible
        if not self.isVisible():
            self.show()
        self.activateWindow()
        self.raise_()

    def add_code_block(self, language: str, code_content: str):
        """Legacy method for adding code blocks from chat"""
        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        lang_display = language.strip().capitalize() if language.strip() else "Code"
        block_name = f"{lang_display} Block ({timestamp})"
        self.update_or_add_file(block_name, code_content, is_ai_modification=False)

    def clear_viewer(self):
        if not self._file_contents:
            return

        response = QMessageBox.question(
            self, "Confirm Clear",
            "Remove all generated files from the viewer?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if response == QMessageBox.StandardButton.Yes:
            self._file_contents.clear()
            self._original_file_contents.clear()
            self.file_tree.clear()
            self.code_edit.clear()
            self.copy_button.setEnabled(False)
            self.apply_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            self._current_filename = None
            self._current_project_id_for_apply = None
            self._current_focus_prefix_for_apply = None
            self._current_content_is_modification = False
            logger.info("Cleared all files from CodeViewerWindow.")

    @Slot(QTreeWidgetItem, QTreeWidgetItem)
    def _display_selected_file_content(self, current_item: Optional[QTreeWidgetItem],
                                       previous_item: Optional[QTreeWidgetItem]):
        self._reset_copy_button_icon()
        self.apply_button.setEnabled(False)
        self._current_content_is_modification = False
        self._current_project_id_for_apply = None
        self._current_focus_prefix_for_apply = None

        if current_item is None:
            self.code_edit.clear()
            self.copy_button.setEnabled(False)
            self._current_filename = None
            return

        # Extract filename (remove emoji prefix if present)
        display_text = current_item.text(0)
        filename = display_text.replace("🤖 ", "") if display_text.startswith("🤖 ") else display_text
        self._current_filename = filename

        code_content = self._file_contents.get(filename)

        if code_content is not None:
            self.code_edit.setPlainText(code_content)
            self.copy_button.setEnabled(True)

            # Enable Apply button if this is an AI modification
            if filename in self._original_file_contents:
                self._current_content_is_modification = True
                # For now, use defaults since we don't have robust context passing
                # This could be improved by storing more metadata per file
                self._current_project_id_for_apply = "p1_chat_context"  # Default for Phase 1
                self._current_focus_prefix_for_apply = constants.APP_BASE_DIR
                self.apply_button.setEnabled(True)

        else:
            logger.warning(f"Content not found for selected file: {filename}")
            self.code_edit.setPlainText(f"[Error: Content not found for {filename}]")
            self.copy_button.setEnabled(False)

    @Slot()
    def _handle_apply_change(self):
        if not self._current_filename or \
                not self._current_content_is_modification or \
                self._current_project_id_for_apply is None:
            logger.warning("Apply change clicked but context is missing or not a modification.")
            QMessageBox.warning(self, "Apply Error",
                                "Cannot apply change: Missing context or not an AI modification.")
            return

        new_content = self._file_contents.get(self._current_filename)
        if new_content is None:
            logger.error(f"Apply change: Content for '{self._current_filename}' is None.")
            QMessageBox.critical(self, "Internal Error", "Content for current file is missing.")
            return

        logger.info(f"Emitting apply_change_requested for: Proj='{self._current_project_id_for_apply}', "
                    f"File='{self._current_filename}', Focus='{self._current_focus_prefix_for_apply}'")

        self.apply_change_requested.emit(
            self._current_project_id_for_apply,
            self._current_filename,
            new_content,
            self._current_focus_prefix_for_apply or ""
        )

        # Provide user feedback
        self.apply_button.setEnabled(False)
        self.apply_button.setText("💾 Applying...")
        QTimer.singleShot(2000, self._reset_apply_button)

    def _reset_apply_button(self):
        self.apply_button.setText("💾 Apply & Save")
        self.apply_button.setEnabled(self._current_content_is_modification)

    def _copy_selected_code_with_feedback(self):
        code_to_copy = self.code_edit.toPlainText()
        if not code_to_copy or self._current_filename is None:
            logger.warning("Attempted to copy empty or unselected code from CodeViewerWindow.")
            return

        if code_to_copy.startswith("[Error:"):
            return

        try:
            clipboard = QApplication.clipboard()
            if not clipboard:
                raise RuntimeError("Clipboard not accessible.")

            clipboard.setText(code_to_copy)
            logger.info(f"Copied code for '{self._current_filename}' from viewer.")

            self.copy_button.setText("✅ Copied!")
            self.copy_button.setEnabled(False)
            QTimer.singleShot(1500, self._reset_copy_button_icon)

        except Exception as e:
            logger.exception(f"Error copying code from viewer: {e}")
            QMessageBox.warning(self, "Copy Error", f"Could not copy code:\n{e}")

    def _reset_copy_button_icon(self):
        self.copy_button.setText("📋 Copy Code")
        self.copy_button.setEnabled(bool(self._current_filename) and self._current_filename in self._file_contents)

    @Slot(str)
    def handle_apply_completed(self, processed_filename: str):
        """Called when a file has been successfully applied"""
        logger.debug(f"CodeViewer: Apply completed for {processed_filename}")
        if self._current_filename == processed_filename:
            self.apply_button.setText("✅ Applied!")
            QTimer.singleShot(2000, self._reset_apply_button)

    def showEvent(self, event):
        super().showEvent(event)
        if self.file_tree.topLevelItemCount() > 0 and self.file_tree.currentItem() is None:
            self.file_tree.setCurrentItem(self.file_tree.topLevelItem(0))
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
# ui/dialogs/project_rag_dialog.py
import logging
import os
from typing import Optional, List

from PySide6.QtCore import Qt, Signal as pyqtSignal, QUrl, Slot
from PySide6.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QDialogButtonBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QFileDialog, QWidget
)

try:
    from utils import constants
    from core.event_bus import EventBus
except ImportError as e_pr_dialog:
    logging.getLogger(__name__).critical(f"Critical import error in ProjectRagDialog: {e_pr_dialog}", exc_info=True)
    # Fallback for constants if needed for basic parsing
    class constants_fallback: #type: ignore
        CHAT_FONT_FAMILY = "Arial"
        CHAT_FONT_SIZE = 10
    constants = constants_fallback #type: ignore
    raise

logger = logging.getLogger(__name__)


class DropTargetWidget(QFrame):
    """
    A QFrame that acts as a drag-and-drop target for files.
    Emits a 'filesDropped' signal with a list of local file paths.
    """
    filesDropped = pyqtSignal(list)  # Signal to emit list of file paths

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.setAcceptDrops(True)
        self.setMinimumHeight(80)
        self.label = QLabel("Drag & Drop Files Here\n(or use 'Browse' button below)", self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("QLabel { color: #909090; font-style: italic; padding: 5px; }")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setStyleSheet("""
            DropTargetWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border: 2px dashed #555;
                border-radius: 8px;
            }
            DropTargetWidget:hover {
                border-color: #00CFE8;
                background-color: rgba(0, 207, 232, 0.07);
            }
        """)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.label.setText("Release to Add Files")
            self.setStyleSheet("""
                DropTargetWidget {
                    background-color: rgba(0, 207, 232, 0.1);
                    border: 2px solid #00CFE8;
                    border-radius: 8px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDropEvent): # Corrected event type
        self.label.setText("Drag & Drop Files Here\n(or use 'Browse' button below)")
        self.setStyleSheet("""
            DropTargetWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border: 2px dashed #555;
                border-radius: 8px;
            }
            DropTargetWidget:hover {
                border-color: #00CFE8;
                background-color: rgba(0, 207, 232, 0.07);
            }
        """)
        event.accept()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        file_paths = []
        if urls:
            for url in urls:
                if url.isLocalFile():
                    file_paths.append(url.toLocalFile())
            if file_paths:
                self.filesDropped.emit(file_paths) # Emit the signal with the list of file paths
            event.acceptProposedAction()
        else:
            event.ignore()
        # Reset label and style after drop
        self.label.setText("Drag & Drop Files Here\n(or use 'Browse' button below)")
        self.setStyleSheet("""
            DropTargetWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border: 2px dashed #555;
                border-radius: 8px;
            }
            DropTargetWidget:hover {
                border-color: #00CFE8;
                background-color: rgba(0, 207, 232, 0.07);
            }
        """)


class ProjectRagDialog(QDialog):
    """
    Dialog for adding files to a specific project's RAG knowledge base.
    Allows file selection via drag-and-drop or a file browser.
    """
    def __init__(self, project_id: str, project_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(f"Add Files to RAG: {project_name}")
        self.setObjectName("ProjectRagDialog")
        self.setMinimumSize(450, 350)
        self.setModal(True)

        self._project_id = project_id
        self._event_bus = EventBus.get_instance()
        self._selected_files: List[str] = []

        self._init_ui()
        self._connect_signals()
        logger.info(f"ProjectRagDialog initialized for project ID: {project_id} ({project_name})")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Info Label
        info_label = QLabel(
            f"Add files to the knowledge base for project: <b>{self.windowTitle().split(': ')[-1]}</b>."
            "\nThese files will be used to provide context for chats within this project."
        )
        info_label.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE -1))
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # Drag and Drop Area
        self.drop_target_widget = DropTargetWidget(self)
        main_layout.addWidget(self.drop_target_widget)

        # Selected Files List (Optional, for user feedback)
        self.selected_files_list_widget = QListWidget()
        self.selected_files_list_widget.setObjectName("ProjectRagSelectedFilesList")
        self.selected_files_list_widget.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE - 1))
        self.selected_files_list_widget.setToolTip("Files staged for adding")
        self.selected_files_list_widget.setFixedHeight(100) # Limit height
        self.selected_files_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        main_layout.addWidget(self.selected_files_list_widget)

        # Browse Button
        self.browse_button = QPushButton("Browse Files...")
        self.browse_button.setFont(QFont(constants.CHAT_FONT_FAMILY, constants.CHAT_FONT_SIZE))
        self.browse_button.setIcon(QIcon.fromTheme("document-open", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-open-16.png"))) # Fallback icon
        main_layout.addWidget(self.browse_button, 0, Qt.AlignmentFlag.AlignLeft)


        # Dialog Buttons (OK/Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Add Selected Files")
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False) # Disabled until files are selected
        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)

    def _connect_signals(self):
        self.drop_target_widget.filesDropped.connect(self._handle_files_selected)
        self.browse_button.clicked.connect(self._browse_for_files)
        self.button_box.accepted.connect(self._on_accept)
        self.button_box.rejected.connect(self.reject)

    def _update_selected_files_display(self):
        self.selected_files_list_widget.clear()
        if not self._selected_files:
            self.selected_files_list_widget.addItem(QListWidgetItem("No files selected."))
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
        else:
            for file_path in self._selected_files:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setToolTip(file_path)
                self.selected_files_list_widget.addItem(item)
            self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)


    @Slot(list)
    def _handle_files_selected(self, file_paths: List[str]):
        # Add to internal list, avoid duplicates, and update display
        newly_added_count = 0
        for fp in file_paths:
            if os.path.isfile(fp) and fp not in self._selected_files: # Ensure it's a file and not already added
                self._selected_files.append(fp)
                newly_added_count +=1
        if newly_added_count > 0:
            self._update_selected_files_display()
            logger.debug(f"ProjectRagDialog: Added {newly_added_count} files via selection/drop. Total: {len(self._selected_files)}")


    @Slot()
    def _browse_for_files(self):
        file_dialog = QFileDialog(self, "Select Files for Project RAG", os.path.expanduser("~"))
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        # Consider using constants.ALLOWED_TEXT_EXTENSIONS for a more specific filter
        allowed_extensions_str = " ".join([f"*{ext}" for ext in constants.ALLOWED_TEXT_EXTENSIONS])
        name_filters = f"Supported Files ({allowed_extensions_str});;All Files (*)"
        file_dialog.setNameFilter(name_filters)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self._handle_files_selected(selected_files)

    @Slot()
    def _on_accept(self):
        if not self._selected_files:
            logger.warning("ProjectRagDialog: 'Add Selected Files' clicked but no files are selected.")
            # Optionally show a message to the user
            return

        logger.info(f"ProjectRagDialog: Emitting requestProjectFilesUpload for {len(self._selected_files)} files, project ID: {self._project_id}")
        self._event_bus.requestProjectFilesUpload.emit(self._selected_files, self._project_id)
        self.accept() # Close the dialog

    def get_selected_files(self) -> List[str]:
        return self._selected_files

    def exec(self) -> int:
        # Reset selected files each time dialog is shown
        self._selected_files = []
        self._update_selected_files_display()
        logger.debug(f"ProjectRagDialog exec() called for project {self._project_id}.")
        return super().exec()


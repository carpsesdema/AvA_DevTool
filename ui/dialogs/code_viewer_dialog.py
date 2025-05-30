# ui/dialogs/code_viewer_dialog.py - Complete Enhanced Multi-Project IDE Version (FIXED)
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Set, Any

from PySide6.QtCore import Qt, QTimer, Slot, Signal, QMimeData, QUrl
from PySide6.QtGui import (
    QIcon, QFontDatabase, QTextOption, QFont, QDragEnterEvent,
    QDropEvent, QKeySequence, QShortcut, QPixmap
)
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QSplitter,
    QApplication, QMessageBox, QTreeWidget, QTreeWidgetItem, QWidget, QLabel,
    QTabWidget, QToolBar, QStatusBar, QLineEdit, QComboBox, QProgressBar,
    QFrame, QStyle, QFileDialog, QInputDialog, QMenu, QHeaderView
)

try:
    from utils import constants
    from utils.constants import CHAT_FONT_FAMILY, CHAT_FONT_SIZE, ALLOWED_TEXT_EXTENSIONS
    from core.event_bus import EventBus
except ImportError as e:
    logging.error(f"Error importing dependencies in enhanced code_viewer_dialog.py: {e}")


    class constants:
        CHAT_FONT_FAMILY = "Arial"
        CHAT_FONT_SIZE = 10
        ALLOWED_TEXT_EXTENSIONS = {'.py', '.txt', '.md', '.json', '.yaml', '.yml'}


    class EventBus:
        @staticmethod
        def get_instance():
            return None

logger = logging.getLogger(__name__)


class ProjectTreeWidget(QTreeWidget):
    """Enhanced tree widget for project file navigation with focus support"""
    fileSelected = Signal(str)  # file_path
    focusRequested = Signal(str, list)  # project_id, file_paths

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setHeaderLabel("Project Files")
        self.setContextMenuPolicy(Qt.CustomContextMenuRequested)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemClicked.connect(self._on_item_clicked)
        self.setAlternatingRowColors(True)
        self.setRootIsDecorated(True)

        # Store current project context
        self._current_project_id: Optional[str] = None
        self._current_project_path: Optional[str] = None

        # File type icons mapping
        self._setup_file_icons()

    def _setup_file_icons(self):
        """Setup file type icons"""
        style = self.style()
        self._icons = {
            'folder': style.standardIcon(QStyle.SP_DirIcon),
            'python': style.standardIcon(QStyle.SP_FileIcon),
            'text': style.standardIcon(QStyle.SP_FileDialogDetailedView),
            'config': style.standardIcon(QStyle.SP_ComputerIcon),
            'document': style.standardIcon(QStyle.SP_FileDialogListView),
            'default': style.standardIcon(QStyle.SP_FileIcon)
        }

    def load_project_structure(self, project_path: str, project_id: Optional[str] = None):
        """Load project files into tree structure"""
        self.clear()
        self._current_project_path = project_path
        self._current_project_id = project_id

        if not os.path.exists(project_path) or not os.path.isdir(project_path):
            logger.error(f"Invalid project path: {project_path}")
            return

        project_name = os.path.basename(project_path)
        root_item = QTreeWidgetItem([project_name])
        root_item.setData(0, Qt.UserRole, project_path)
        root_item.setData(1, Qt.UserRole, 'folder')
        root_item.setIcon(0, self._icons['folder'])
        self.addTopLevelItem(root_item)

        try:
            self._populate_tree_recursive(root_item, project_path)
            root_item.setExpanded(True)
            logger.info(f"Loaded project structure for: {project_name}")
        except Exception as e:
            logger.error(f"Error loading project structure: {e}")

    def _populate_tree_recursive(self, parent_item: QTreeWidgetItem, dir_path: str):
        """Recursively populate tree with files and folders"""
        try:
            items = []
            for item_name in os.listdir(dir_path):
                if item_name.startswith('.'):  # Skip hidden files
                    continue

                item_path = os.path.join(dir_path, item_name)
                items.append((item_name, item_path))

            # Sort: directories first, then files
            items.sort(key=lambda x: (not os.path.isdir(x[1]), x[0].lower()))

            for item_name, item_path in items:
                tree_item = QTreeWidgetItem([item_name])
                tree_item.setData(0, Qt.UserRole, item_path)

                if os.path.isdir(item_path):
                    # Directory
                    tree_item.setData(1, Qt.UserRole, 'folder')
                    tree_item.setIcon(0, self._icons['folder'])
                    parent_item.addChild(tree_item)

                    # Skip common ignored directories
                    if item_name.lower() not in {'__pycache__', '.git', 'node_modules', '.vscode', '.idea'}:
                        self._populate_tree_recursive(tree_item, item_path)
                else:
                    # File
                    file_ext = os.path.splitext(item_name)[1].lower()
                    if file_ext in constants.ALLOWED_TEXT_EXTENSIONS:
                        tree_item.setData(1, Qt.UserRole, 'file')
                        tree_item.setIcon(0, self._get_file_icon(item_name))
                        parent_item.addChild(tree_item)

        except PermissionError:
            logger.warning(f"Permission denied accessing: {dir_path}")
        except Exception as e:
            logger.error(f"Error populating tree for {dir_path}: {e}")

    def _get_file_icon(self, filename: str) -> QIcon:
        """Get appropriate icon for file type"""
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.py':
            return self._icons['python']
        elif ext in {'.txt', '.md', '.rst'}:
            return self._icons['document']
        elif ext in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}:
            return self._icons['config']
        else:
            return self._icons['default']

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle item click - emit file selection for files"""
        item_path = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)

        if item_type == 'file' and os.path.isfile(item_path):
            self.fileSelected.emit(item_path)

    def _show_context_menu(self, position):
        """Show context menu for tree items"""
        item = self.itemAt(position)
        if not item:
            return

        menu = QMenu(self)
        item_path = item.data(0, Qt.UserRole)
        item_type = item.data(1, Qt.UserRole)

        if item_type == 'file':
            # File actions
            focus_action = menu.addAction("🎯 Set Focus on File")
            focus_action.triggered.connect(lambda: self._set_focus_on_files([item_path]))

            menu.addSeparator()
            open_action = menu.addAction("📁 Open in System")
            open_action.triggered.connect(lambda: self._open_in_system(item_path))

        elif item_type == 'folder':
            # Folder actions
            focus_action = menu.addAction("🎯 Set Focus on Folder")
            focus_action.triggered.connect(lambda: self._set_focus_on_folder(item_path))

            menu.addSeparator()
            open_action = menu.addAction("📁 Open in System")
            open_action.triggered.connect(lambda: self._open_in_system(item_path))

        if menu.actions():
            menu.exec(self.mapToGlobal(position))

    def _set_focus_on_files(self, file_paths: List[str]):
        """Set focus on specific files"""
        if self._current_project_id and file_paths:
            self.focusRequested.emit(self._current_project_id, file_paths)
            logger.info(f"Focus requested on {len(file_paths)} files")

    def _set_focus_on_folder(self, folder_path: str):
        """Set focus on all Python files in a folder"""
        if not self._current_project_id:
            return

        py_files = []
        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.py'):
                        py_files.append(os.path.join(root, file))
                # Limit depth to avoid huge focus sets
                if len(py_files) > 50:
                    break

            if py_files:
                self.focusRequested.emit(self._current_project_id, py_files)
                logger.info(f"Focus requested on {len(py_files)} Python files in folder")
            else:
                QMessageBox.information(self, "Focus", "No Python files found in this folder.")

        except Exception as e:
            logger.error(f"Error setting focus on folder: {e}")

    def _open_in_system(self, path: str):
        """Open file/folder in system file manager"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(path)
            elif os.name == 'posix':  # macOS/Linux
                import subprocess
                if os.uname().sysname == 'Darwin':  # macOS
                    subprocess.call(['open', path])
                else:  # Linux
                    subprocess.call(['xdg-open', path])
        except Exception as e:
            logger.error(f"Error opening in system: {e}")


class CodeEditorWidget(QTextEdit):
    """Enhanced code editor with syntax highlighting and save capability"""
    fileSaved = Signal(str, str, str)  # project_id, file_path, content
    contentModified = Signal(bool)  # is_modified

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current_file_path: Optional[str] = None
        self._current_project_id: Optional[str] = None
        self._is_modified = False
        self._original_content = ""

        self._setup_editor()

    def _setup_editor(self):
        """Setup editor properties"""
        # Font setup
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(constants.CHAT_FONT_SIZE)
        self.setFont(font)

        # Editor properties
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.setWordWrapMode(QTextOption.NoWrap)
        self.setTabStopDistance(40)  # 4 spaces

        # Connect change detection
        self.textChanged.connect(self._on_content_changed)

        # Keyboard shortcuts
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.save_file)

    def load_file(self, file_path: str, project_id: Optional[str] = None):
        """Load file content into editor"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            self.setPlainText(content)
            self._current_file_path = file_path
            self._current_project_id = project_id
            self._original_content = content
            self._is_modified = False
            self._update_modified_state()

            logger.info(f"Loaded file: {os.path.basename(file_path)}")

        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            QMessageBox.warning(self, "Error", f"Could not load file: {e}")

    def save_file(self) -> bool:
        """Save current content to file"""
        if not self._current_file_path:
            return False

        try:
            content = self.toPlainText()

            # Create backup if file exists
            if os.path.exists(self._current_file_path):
                backup_path = f"{self._current_file_path}.backup"
                with open(self._current_file_path, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as b:
                        b.write(f.read())

            # Write new content
            with open(self._current_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self._original_content = content
            self._is_modified = False
            self._update_modified_state()

            # Emit save signal for RAG sync
            if self._current_project_id:
                self.fileSaved.emit(self._current_project_id, self._current_file_path, content)

            logger.info(f"Saved file: {os.path.basename(self._current_file_path)}")
            return True

        except Exception as e:
            logger.error(f"Error saving file: {e}")
            QMessageBox.warning(self, "Error", f"Could not save file: {e}")
            return False

    def _on_content_changed(self):
        """Handle content changes"""
        if self._current_file_path:
            current_content = self.toPlainText()
            was_modified = self._is_modified
            self._is_modified = (current_content != self._original_content)

            if was_modified != self._is_modified:
                self._update_modified_state()

    def _update_modified_state(self):
        """Update UI to reflect modification state"""
        self.contentModified.emit(self._is_modified)

    def get_current_file_info(self) -> tuple:
        """Get current file info (path, project_id, is_modified)"""
        return (self._current_file_path, self._current_project_id, self._is_modified)


class CodeViewerWindow(QDialog):
    """Enhanced multi-project IDE window"""

    # Signals
    apply_change_requested = Signal(str, str, str, str)  # project_id, relative_filepath, new_content, focus_prefix
    projectFilesSaved = Signal(str, str, str)  # project_id, file_path, content
    focusSetOnFiles = Signal(str, list)  # project_id, file_paths

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("AvA Code Viewer - IDE Mode")
        self.setObjectName("CodeViewerWindow")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        self.setModal(False)
        self.setAcceptDrops(True)

        # Multi-project state
        self._open_projects: Dict[str, Dict[str, Any]] = {}  # project_id -> project_info
        self._current_project_tab: Optional[int] = None

        # Legacy single-file state (for backward compatibility)
        self._file_contents: Dict[str, str] = {}
        self._original_file_contents: Dict[str, Optional[str]] = {}
        self._current_filename: Optional[str] = None
        self._current_project_id_for_apply: Optional[str] = None
        self._current_focus_prefix_for_apply: Optional[str] = None
        self._current_content_is_modification: bool = False

        # Initialize UI
        self._init_enhanced_ui()
        self._connect_signals()

        # Get event bus reference
        self._event_bus = EventBus.get_instance()

        logger.info("Enhanced CodeViewerWindow initialized")

    def _init_enhanced_ui(self):
        """Initialize the enhanced IDE interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Toolbar
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)

        # Main content area
        self.project_tabs = QTabWidget()
        self.project_tabs.setTabsClosable(True)
        self.project_tabs.tabCloseRequested.connect(self._close_project_tab)
        self.project_tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.project_tabs, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Ready - Drop project folder or use File menu")
        self.status_bar.addWidget(self.status_label)
        layout.addWidget(self.status_bar)

        # Create default tab for generated files (backward compatibility)
        self._create_generated_files_tab()

    def _create_toolbar(self) -> QToolBar:
        """Create toolbar with project actions"""
        toolbar = QToolBar()

        # Project actions
        toolbar.addAction("📂 Open Project", self._open_project_dialog)
        toolbar.addAction("💾 Save All", self._save_all_files)
        toolbar.addSeparator()

        # View actions
        toolbar.addAction("🔍 Find in Files", self._find_in_files)
        toolbar.addAction("🎯 Clear Focus", self._clear_all_focus)
        toolbar.addSeparator()

        # Legacy actions (for backward compatibility)
        toolbar.addAction("📄 Clear Generated", self._clear_generated_files)
        toolbar.addSeparator()

        # Help
        toolbar.addAction("❓ Help", self._show_help)

        return toolbar

    def _create_generated_files_tab(self):
        """Create the traditional generated files tab for backward compatibility"""
        tab_widget = QWidget()
        layout = QHBoxLayout(tab_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: File tree for generated files
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Generated files label
        generated_label = QLabel("🤖 Generated Files")
        generated_label.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(generated_label)

        # File tree (simplified for generated files)
        self.generated_files_tree = QTreeWidget()
        self.generated_files_tree.setHeaderLabel("Generated Files")
        self.generated_files_tree.itemClicked.connect(self._on_generated_file_clicked)
        left_layout.addWidget(self.generated_files_tree)

        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)

        # Right panel: Code display and actions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # File info and actions
        actions_layout = QHBoxLayout()
        self.current_file_label = QLabel("No file selected")
        actions_layout.addWidget(self.current_file_label)
        actions_layout.addStretch()

        # Action buttons
        self.copy_button = QPushButton("📋 Copy")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_selected_code_with_feedback)
        actions_layout.addWidget(self.copy_button)

        self.apply_button = QPushButton("💾 Apply")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._handle_apply_change)
        actions_layout.addWidget(self.apply_button)

        right_layout.addLayout(actions_layout)

        # Code display
        self.code_edit = QTextEdit()
        self.code_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.code_edit.setReadOnly(True)
        right_layout.addWidget(self.code_edit)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)

        self.project_tabs.addTab(tab_widget, "Generated Files")

    def _connect_signals(self):
        """Connect UI signals"""
        pass  # Individual components handle their own signals

    # Drag and Drop Support
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                path = Path(urls[0].toLocalFile())
                if path.is_dir():
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop events"""
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if path.is_dir():
                self._load_project_from_path(str(path))
                event.acceptProposedAction()
            else:
                event.ignore()

    def _load_project_from_path(self, project_path: str):
        """Load a project from file system path"""
        if not os.path.exists(project_path) or not os.path.isdir(project_path):
            QMessageBox.warning(self, "Invalid Project", "Please select a valid directory.")
            return

        project_name = os.path.basename(project_path)

        # Generate project ID (simplified for now)
        project_id = f"proj_{project_name.lower().replace(' ', '_').replace('-', '_')}"

        # Check if already open
        for i in range(self.project_tabs.count()):
            if self.project_tabs.tabText(i) == project_name:
                self.project_tabs.setCurrentIndex(i)
                return

        # Create new project tab
        tab_widget = self._create_project_tab_widget(project_name, project_path, project_id)
        tab_index = self.project_tabs.addTab(tab_widget, project_name)
        self.project_tabs.setCurrentIndex(tab_index)

        # Store project info
        self._open_projects[project_id] = {
            'name': project_name,
            'path': project_path,
            'tab_index': tab_index
        }

        self.status_label.setText(f"Loaded project: {project_name}")
        logger.info(f"Loaded project: {project_name} from {project_path}")

        # Show window if hidden
        if not self.isVisible():
            self.show()
        self.activateWindow()
        self.raise_()

    def _create_project_tab_widget(self, project_name: str, project_path: Optional[str],
                                   project_id: Optional[str]) -> QWidget:
        """Create a tab widget for a project"""
        tab_widget = QWidget()
        layout = QHBoxLayout(tab_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: Project tree
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # Project info
        project_label = QLabel(f"📁 {project_name}")
        project_label.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(project_label)

        # Project tree
        project_tree = ProjectTreeWidget()
        project_tree.focusRequested.connect(self._handle_focus_request)
        project_tree.fileSelected.connect(self._load_file_in_editor)
        left_layout.addWidget(project_tree)

        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)

        # Right panel: Code editor
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)

        # File info bar
        file_info_layout = QHBoxLayout()
        file_path_label = QLabel("No file selected")
        file_path_label.setObjectName("filePathLabel")
        file_info_layout.addWidget(file_path_label)
        file_info_layout.addStretch()

        # Editor controls
        save_button = QPushButton("💾 Save")
        save_button.setObjectName("saveButton")
        save_button.setEnabled(False)
        file_info_layout.addWidget(save_button)

        right_layout.addLayout(file_info_layout)

        # Code editor
        code_editor = CodeEditorWidget()
        code_editor.contentModified.connect(lambda modified: save_button.setEnabled(modified))
        code_editor.fileSaved.connect(self._handle_file_saved)
        right_layout.addWidget(code_editor)

        # Connect save button
        save_button.clicked.connect(code_editor.save_file)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)

        # Load project if path provided
        if project_path and project_id:
            project_tree.load_project_structure(project_path, project_id)

        return tab_widget

    def _handle_focus_request(self, project_id: str, file_paths: List[str]):
        """Handle focus requests from project tree"""
        self.focusSetOnFiles.emit(project_id, file_paths)
        self.status_label.setText(f"Focus set on {len(file_paths)} files")
        logger.info(f"Focus set on {len(file_paths)} files for project {project_id}")

    def _load_file_in_editor(self, file_path: str):
        """Load a file into the current tab's editor"""
        current_tab = self.project_tabs.currentWidget()
        if not current_tab:
            return

        editor = current_tab.findChild(CodeEditorWidget)
        file_label = current_tab.findChild(QLabel, "filePathLabel")

        if editor and file_label:
            # Determine project ID from current tab
            project_id = None
            for pid, info in self._open_projects.items():
                if info.get('tab_index') == self.project_tabs.currentIndex():
                    project_id = pid
                    break

            editor.load_file(file_path, project_id)
            file_label.setText(os.path.basename(file_path))

            # Update window title
            self.setWindowTitle(f"AvA Code Viewer - {os.path.basename(file_path)}")

    def _handle_file_saved(self, project_id: str, file_path: str, content: str):
        """Handle file save events"""
        self.projectFilesSaved.emit(project_id, file_path, content)
        self.status_label.setText(f"Saved: {os.path.basename(file_path)}")
        logger.info(f"File saved: {file_path}")

    # Generated Files Tab Methods (Legacy Support)
    def _on_generated_file_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle click on generated file"""
        filename = item.text(0)
        # Remove AI prefix if present
        if filename.startswith("🤖 "):
            filename = filename[2:]

        if filename in self._file_contents:
            self._display_generated_file_content(filename)

    def _display_generated_file_content(self, filename: str):
        """Display content of a generated file"""
        content = self._file_contents.get(filename, "")
        self.code_edit.setPlainText(content)
        self.current_file_label.setText(filename)
        self._current_filename = filename

        # Enable/disable buttons
        self.copy_button.setEnabled(bool(content))

        # Check if this is a modification
        is_modification = filename in self._original_file_contents
        self._current_content_is_modification = is_modification
        self.apply_button.setEnabled(is_modification)

        # Update window title
        self.setWindowTitle(f"AvA Code Viewer - {filename}")

    # Toolbar Actions
    def _open_project_dialog(self):
        """Open project selection dialog"""
        project_path = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", os.getcwd()
        )
        if project_path:
            self._load_project_from_path(project_path)

    def _save_all_files(self):
        """Save all modified files"""
        saved_count = 0
        for i in range(1, self.project_tabs.count()):  # Skip generated files tab
            tab_widget = self.project_tabs.widget(i)
            editor = tab_widget.findChild(CodeEditorWidget)
            if editor:
                file_path, project_id, is_modified = editor.get_current_file_info()
                if is_modified and editor.save_file():
                    saved_count += 1

        if saved_count > 0:
            self.status_label.setText(f"Saved {saved_count} files")
        else:
            self.status_label.setText("No files to save")

    def _find_in_files(self):
        """Find in files functionality (placeholder)"""
        QMessageBox.information(self, "Find in Files", "Find in Files functionality coming soon!")

    def _clear_all_focus(self):
        """Clear all focus settings"""
        # This would emit a signal to clear focus in the RAG system
        self.status_label.setText("All focus cleared")

    def _clear_generated_files(self):
        """Clear all generated files"""
        reply = QMessageBox.question(
            self, "Clear Generated Files",
            "This will clear all generated files. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.clear_viewer()

    def _show_help(self):
        """Show help dialog"""
        help_text = """
AvA Code Viewer - IDE Mode

Features:
• Drag & drop project folders to load them
• Navigate project files in the tree view
• Edit files directly with auto-save (Ctrl+S)
• Right-click files/folders to set AI focus
• Multiple projects in tabs
• Automatic RAG synchronization on save
• Generated files tab for AI-created code

Keyboard Shortcuts:
• Ctrl+S: Save current file
• Ctrl+O: Open project

Focus System:
Right-click on files or folders to set focus for AI context.
This helps the AI understand which parts of your project are most relevant.

Generated Files:
The first tab shows AI-generated code files.
Use Apply button to save modifications to your project.
        """
        QMessageBox.information(self, "Help", help_text.strip())

    def _close_project_tab(self, index: int):
        """Close a project tab"""
        if index == 0:  # Don't close generated files tab
            return

        tab_text = self.project_tabs.tabText(index)
        reply = QMessageBox.question(
            self, "Close Project",
            f"Close project '{tab_text}'?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Remove from open projects
            project_to_remove = None
            for project_id, info in self._open_projects.items():
                if info.get('tab_index') == index:
                    project_to_remove = project_id
                    break

            if project_to_remove:
                del self._open_projects[project_to_remove]

                # Update tab indices for remaining projects
                for info in self._open_projects.values():
                    if info.get('tab_index', 0) > index:
                        info['tab_index'] -= 1

            self.project_tabs.removeTab(index)
            self.status_label.setText(f"Closed project: {tab_text}")

    def _on_tab_changed(self, index: int):
        """Handle tab change"""
        if index >= 0:
            tab_text = self.project_tabs.tabText(index)
            self.setWindowTitle(f"AvA Code Viewer - {tab_text}")

    # Legacy Methods (for backward compatibility with existing AI code generation)
    def update_or_add_file(self, filename: str, content: str, is_ai_modification: bool = False,
                           original_content: Optional[str] = None, project_id_for_apply: Optional[str] = None,
                           focus_prefix_for_apply: Optional[str] = None):
        """Legacy method for backward compatibility with existing AI code generation"""
        logger.info(f"Legacy file add: {filename}")

        # Store in legacy file contents
        self._file_contents[filename] = content
        if is_ai_modification:
            self._original_file_contents[filename] = original_content
            self._current_project_id_for_apply = project_id_for_apply
            self._current_focus_prefix_for_apply = focus_prefix_for_apply

        # Add to Generated Files tab tree
        tree = self.generated_files_tree

        # Check if file already exists in tree
        existing_item = None
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            item_name = item.text(0)
            # Remove AI prefix for comparison
            if item_name.startswith("🤖 "):
                item_name = item_name[2:]
            if item_name == filename:
                existing_item = item
                break

        if existing_item:
            # Update existing item
            if is_ai_modification:
                existing_item.setText(0, f"🤖 {filename}")
            else:
                existing_item.setText(0, filename)
        else:
            # Create new item
            display_name = f"🤖 {filename}" if is_ai_modification else filename
            new_item = QTreeWidgetItem([display_name])
            tree.addTopLevelItem(new_item)

        # Select the item and display content
        if existing_item:
            tree.setCurrentItem(existing_item)
        else:
            tree.setCurrentItem(tree.topLevelItem(tree.topLevelItemCount() - 1))

        self._display_generated_file_content(filename)

        # Show window if hidden
        if not self.isVisible():
            self.show()
        self.activateWindow()
        self.raise_()

        # Switch to generated files tab
        self.project_tabs.setCurrentIndex(0)

    def add_code_block(self, language: str, code_content: str):
        """Legacy method for adding arbitrary code snippets"""
        timestamp = datetime.now().strftime("%H%M%S")
        filename = f"{language}_snippet_{timestamp}.{language}"
        self.update_or_add_file(filename, code_content)

    def clear_viewer(self):
        """Clear all generated files"""
        reply = QMessageBox.question(
            self, "Clear Viewer",
            "This will clear all generated files. Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Clear data
            self._file_contents.clear()
            self._original_file_contents.clear()
            self._current_filename = None
            self._current_project_id_for_apply = None
            self._current_focus_prefix_for_apply = None
            self._current_content_is_modification = False

            # Clear UI
            self.generated_files_tree.clear()
            self.code_edit.clear()
            self.current_file_label.setText("No file selected")
            self.copy_button.setEnabled(False)
            self.apply_button.setEnabled(False)

            self.status_label.setText("Generated files cleared")

    def _copy_selected_code_with_feedback(self):
        """Copy the code_edit's text to clipboard with visual feedback"""
        content = self.code_edit.toPlainText()
        if content:
            clipboard = QApplication.clipboard()
            clipboard.setText(content)

            # Visual feedback
            original_text = self.copy_button.text()
            self.copy_button.setText("✅ Copied!")
            self.copy_button.setEnabled(False)

            # Reset after delay
            QTimer.singleShot(1500, lambda: self._reset_copy_button(original_text))

    def _reset_copy_button(self, original_text: str):
        """Reset copy button text and state"""
        self.copy_button.setText(original_text)
        self.copy_button.setEnabled(True)

    def _handle_apply_change(self):
        """Handle apply change for AI modifications"""
        if not self._current_filename or not self._current_content_is_modification:
            return

        content = self.code_edit.toPlainText()
        if not content.strip():
            QMessageBox.warning(self, "Apply Error", "No content to apply.")
            return

        # Emit apply change signal
        self.apply_change_requested.emit(
            self._current_project_id_for_apply or "",
            self._current_filename,
            content,
            self._current_focus_prefix_for_apply or ""
        )

        # Visual feedback
        original_text = self.apply_button.text()
        self.apply_button.setText("✅ Applied!")
        self.apply_button.setEnabled(False)

        # Reset after delay
        QTimer.singleShot(2000, lambda: self._reset_apply_button(original_text))

    def _reset_apply_button(self, original_text: str):
        """Reset apply button text and state"""
        self.apply_button.setText(original_text)
        if self._current_content_is_modification:
            self.apply_button.setEnabled(True)

    def handle_apply_completed(self, processed_filename: str):
        """Handle successful file application (external callback)"""
        if processed_filename == self._current_filename:
            self.status_label.setText(f"Applied: {processed_filename}")

    # Window Events
    def showEvent(self, event):
        """Handle show event"""
        super().showEvent(event)
        # Select first file if none selected in generated files tab
        if (self.project_tabs.currentIndex() == 0 and
                self.generated_files_tree.topLevelItemCount() > 0 and
                not self.generated_files_tree.currentItem()):
            self.generated_files_tree.setCurrentItem(self.generated_files_tree.topLevelItem(0))

    def closeEvent(self, event):
        """Handle close event - hide instead of close to preserve state"""
        self.hide()
        event.ignore()

    # Public API for external integration
    def get_or_create_code_viewer(self):
        """Get this code viewer instance (for compatibility)"""
        return self

    def display_file_in_code_viewer(self, filename: str, content: str,
                                    project_id: Optional[str] = None,
                                    focus_prefix: Optional[str] = None) -> bool:
        """Display file content (for external integration)"""
        try:
            self.update_or_add_file(
                filename, content,
                is_ai_modification=True,
                project_id_for_apply=project_id,
                focus_prefix_for_apply=focus_prefix
            )
            return True
        except Exception as e:
            logger.error(f"Error displaying file: {e}")
            return False
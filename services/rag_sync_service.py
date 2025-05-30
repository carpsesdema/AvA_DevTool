# services/rag_sync_service.py - New file for handling project-specific RAG sync
import logging
import os
import asyncio
from typing import Optional, Dict, Set
from pathlib import Path

from PySide6.QtCore import QObject, Slot, QTimer
from core.event_bus import EventBus

try:
    from services.upload_service import UploadService
    from services.vector_db_service import VectorDBService
    from services.project_service import ProjectManager
except ImportError as e:
    logging.error(f"RagSyncService import error: {e}")

logger = logging.getLogger(__name__)


class RagSyncService(QObject):
    """Service for synchronizing project files with RAG collections in real-time"""

    def __init__(self, upload_service: UploadService, project_manager: ProjectManager, parent=None):
        super().__init__(parent)
        self._upload_service = upload_service
        self._project_manager = project_manager
        self._event_bus = EventBus.get_instance()

        # Track sync state
        self._sync_queue: Dict[str, Set[str]] = {}  # project_id -> set of pending file paths
        self._sync_timer = QTimer()
        self._sync_timer.setSingleShot(True)
        self._sync_timer.timeout.connect(self._process_sync_queue)

        # Connect to events
        self._connect_signals()

        logger.info("RagSyncService initialized")

    def _connect_signals(self):
        """Connect to EventBus signals"""
        if self._event_bus:
            self._event_bus.projectFilesSaved.connect(self._handle_file_saved)
            self._event_bus.projectLoaded.connect(self._handle_project_loaded)
            self._event_bus.ragProjectInitializationRequested.connect(self._handle_project_initialization)

    @Slot(str, str, str)
    def _handle_file_saved(self, project_id: str, file_path: str, content: str):
        """Handle file save events from CodeViewer"""
        logger.info(f"RAG sync requested for {project_id}: {os.path.basename(file_path)}")

        # Add to sync queue
        if project_id not in self._sync_queue:
            self._sync_queue[project_id] = set()
        self._sync_queue[project_id].add(file_path)

        # Debounce sync operations (wait 2 seconds for more changes)
        self._sync_timer.stop()
        self._sync_timer.start(2000)

        # Emit immediate feedback
        if self._event_bus:
            self._event_bus.uiStatusUpdateGlobal.emit(
                f"Queued RAG sync: {os.path.basename(file_path)}",
                "#e5c07b", True, 3000
            )

    @Slot(str, str)
    def _handle_project_loaded(self, project_id: str, project_path: str):
        """Handle project loading - offer to initialize RAG"""
        logger.info(f"Project loaded: {project_id} at {project_path}")

        # Check if project has existing RAG collection
        if self._upload_service and self._upload_service.is_vector_db_ready(project_id):
            logger.info(f"RAG collection exists for project {project_id}")
            if self._event_bus:
                self._event_bus.uiStatusUpdateGlobal.emit(
                    f"Project knowledge ready: {os.path.basename(project_path)}",
                    "#4ade80", True, 4000
                )
        else:
            logger.info(f"No RAG collection for project {project_id} - auto-initializing")
            # Auto-initialize RAG for new projects
            if self._event_bus:
                self._event_bus.ragProjectInitializationRequested.emit(project_id, project_path)

    @Slot(str, str)
    def _handle_project_initialization(self, project_id: str, project_path: str):
        """Handle full project RAG initialization"""
        logger.info(f"Initializing RAG for project {project_id}")

        if self._event_bus:
            self._event_bus.showLoader.emit("Initializing project knowledge base...")

        # Run initialization in background
        asyncio.create_task(self._initialize_project_rag_async(project_id, project_path))

    async def _initialize_project_rag_async(self, project_id: str, project_path: str):
        """Asynchronously initialize RAG for entire project"""
        try:
            if not self._upload_service:
                raise Exception("UploadService not available")

            # Wait for embedder if needed
            if not await self._upload_service.wait_for_embedder_ready():
                raise Exception("RAG embedder not ready")

            # Process entire project directory
            result = self._upload_service.process_directory_for_context(project_path, project_id)

            if result and result.role != "error":
                success_msg = f"Project RAG initialized: {project_id}"
                logger.info(success_msg)

                if self._event_bus:
                    self._event_bus.hideLoader.emit()
                    self._event_bus.uiStatusUpdateGlobal.emit(
                        "Project knowledge base ready ✓", "#4ade80", True, 5000
                    )
            else:
                error_msg = f"Failed to initialize RAG for {project_id}"
                logger.error(error_msg)

                if self._event_bus:
                    self._event_bus.hideLoader.emit()
                    self._event_bus.uiErrorGlobal.emit(error_msg, False)

        except Exception as e:
            logger.error(f"Error initializing project RAG: {e}")
            if self._event_bus:
                self._event_bus.hideLoader.emit()
                self._event_bus.uiErrorGlobal.emit(f"RAG initialization failed: {e}", False)

    def _process_sync_queue(self):
        """Process queued file sync operations"""
        if not self._sync_queue:
            return

        logger.info(f"Processing RAG sync queue: {len(self._sync_queue)} projects")

        for project_id, file_paths in self._sync_queue.items():
            asyncio.create_task(self._sync_files_async(project_id, list(file_paths)))

        # Clear queue
        self._sync_queue.clear()

    async def _sync_files_async(self, project_id: str, file_paths: list):
        """Asynchronously sync files to RAG collection"""
        try:
            if not self._upload_service:
                logger.error("UploadService not available for RAG sync")
                return

            logger.info(f"Syncing {len(file_paths)} files to RAG collection {project_id}")

            if self._event_bus:
                self._event_bus.uiStatusUpdateGlobal.emit(
                    f"Updating knowledge: {len(file_paths)} files",
                    "#61dafb", False, 0
                )

            # Remove old chunks for updated files
            vector_db = getattr(self._upload_service, '_vector_db_service', None)
            if vector_db:
                for file_path in file_paths:
                    vector_db.remove_document_chunks_by_source(project_id, file_path)
                    logger.debug(f"Removed old chunks for {file_path}")

            # Process updated files
            result = await self._upload_service.process_files_for_context_async(file_paths, project_id)

            if result and result.role != "error":
                success_msg = f"RAG updated: {len(file_paths)} files"
                logger.info(success_msg)

                if self._event_bus:
                    self._event_bus.uiStatusUpdateGlobal.emit(
                        "Knowledge updated ✓", "#4ade80", True, 3000
                    )

                    # Emit completion signals for each file
                    for file_path in file_paths:
                        self._event_bus.ragProjectSyncCompleted.emit(project_id, file_path, True)
            else:
                error_msg = f"Failed to sync files to RAG: {project_id}"
                logger.error(error_msg)

                if self._event_bus:
                    self._event_bus.uiStatusUpdateGlobal.emit(
                        "Knowledge sync failed", "#ef4444", True, 5000
                    )

                    # Emit failure signals
                    for file_path in file_paths:
                        self._event_bus.ragProjectSyncCompleted.emit(project_id, file_path, False)

        except Exception as e:
            logger.error(f"Error syncing files to RAG: {e}")
            if self._event_bus:
                self._event_bus.uiStatusUpdateGlobal.emit(
                    f"RAG sync error: {e}", "#ef4444", True, 5000
                )

    def get_sync_status(self, project_id: str) -> dict:
        """Get sync status for a project"""
        return {
            'project_id': project_id,
            'pending_files': len(self._sync_queue.get(project_id, set())),
            'is_syncing': self._sync_timer.isActive()
        }

    def request_manual_sync(self, project_id: str, project_path: str):
        """Manually request full project sync"""
        if self._event_bus:
            self._event_bus.ragProjectInitializationRequested.emit(project_id, project_path)
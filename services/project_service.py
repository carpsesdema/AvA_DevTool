# services/project_service.py
import json
import logging
import os
import shutil  # Added for file backup and removal
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

from PySide6.QtCore import QObject, Signal

try:
    from core.models import ChatMessage
    from core.message_enums import MessageLoadingState  # Added import for enum handling
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in ProjectService: {e}", exc_info=True)
    # Fallback for type hinting if ChatMessage is not available at this point
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    raise

logger = logging.getLogger(__name__)


@dataclass
class Project:
    id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    current_session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        return cls(**data)


@dataclass
class ChatSession:
    id: str
    project_id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    message_history: List[ChatMessage] = field(default_factory=list)  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure ChatMessage objects are correctly serialized
        message_history_serialized = []
        for msg in self.message_history:
            msg_dict = msg.to_dict() if hasattr(msg, 'to_dict') else asdict(msg)

            # Fix for MessageLoadingState serialization
            if 'loading_state' in msg_dict:
                # Convert enum to string representation
                if hasattr(msg_dict['loading_state'], 'name'):
                    msg_dict['loading_state'] = msg_dict['loading_state'].name
                elif hasattr(msg_dict['loading_state'], 'value'):
                    msg_dict['loading_state'] = msg_dict['loading_state'].value
                # If it's not an expected type, just convert to string
                elif msg_dict['loading_state'] is not None:
                    msg_dict['loading_state'] = str(msg_dict['loading_state'])

            message_history_serialized.append(msg_dict)

        data['message_history'] = message_history_serialized
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        messages_data = data.get('message_history', [])
        messages = []
        for msg_data in messages_data:
            if isinstance(msg_data, ChatMessage):  # type: ignore
                messages.append(msg_data)
            elif isinstance(msg_data, dict):
                try:
                    # Convert string loading_state back to enum if needed
                    if 'loading_state' in msg_data and isinstance(msg_data['loading_state'], str):
                        try:
                            from core.message_enums import MessageLoadingState
                            msg_data['loading_state'] = MessageLoadingState[msg_data['loading_state']]
                        except (KeyError, ImportError):
                            # If enum can't be restored, remove it to use default
                            msg_data.pop('loading_state', None)

                    # Assuming ChatMessage can be initialized from a dict (e.g., via **kwargs in its __init__)
                    messages.append(ChatMessage(**msg_data))  # type: ignore
                except Exception as e_msg_create:
                    logger.error(f"Error creating ChatMessage from dict: {msg_data}, error: {e_msg_create}")
            else:
                logger.warning(f"Skipping unknown message data type in ChatSession.from_dict: {type(msg_data)}")

        session_data = data.copy()
        session_data['message_history'] = messages
        return cls(**session_data)


class ProjectManager(QObject):
    projectCreated = Signal(str)
    projectSwitched = Signal(str)
    sessionCreated = Signal(str, str)
    sessionSwitched = Signal(str, str)
    projectsLoaded = Signal(list)  # List[Project]
    projectDeleted = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.projects_dir = os.path.join(constants.USER_DATA_DIR, "projects")
        self.projects_index_file = os.path.join(self.projects_dir, "projects_index.json")
        self._current_project: Optional[Project] = None
        self._current_session: Optional[ChatSession] = None
        self._projects_cache: Dict[str, Project] = {}
        self._sessions_cache: Dict[str, ChatSession] = {}
        self._ensure_directories()
        self._load_projects_index()
        logger.info("ProjectManager initialized.")

    def _ensure_directories(self):
        try:
            os.makedirs(self.projects_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create projects directory: {e}")
            raise

    def _load_projects_index(self):
        if not os.path.exists(self.projects_index_file):
            return
        try:
            with open(self.projects_index_file, 'r', encoding='utf-8') as f:
                projects_data = json.load(f)
            self._projects_cache = {pid: Project.from_dict(data) for pid, data in projects_data.items()}
            self.projectsLoaded.emit(list(self._projects_cache.values()))
        except Exception as e:
            logger.error(f"Failed to load projects index: {e}")
            self._projects_cache = {}

    def _save_projects_index(self):
        try:
            with open(self.projects_index_file, 'w', encoding='utf-8') as f:
                json.dump({pid: p.to_dict() for pid, p in self._projects_cache.items()}, f, indent=2,
                          ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save projects index: {e}")

    def create_project(self, name: str, description: str = "") -> Project:
        project_id = str(uuid.uuid4())
        project = Project(id=project_id, name=name, description=description)

        # ***** FIX: Add project to cache BEFORE creating its default session *****
        self._projects_cache[project_id] = project

        project_dir = os.path.join(self.projects_dir, project_id)
        try:
            os.makedirs(project_dir, exist_ok=True)
            os.makedirs(os.path.join(project_dir, "sessions"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "generated_files"), exist_ok=True)
        except OSError as e:
            logger.error(f"Error creating directories for project {project_id}: {e}")
            # Clean up by removing from cache if directory creation failed significantly
            if project_id in self._projects_cache:
                del self._projects_cache[project_id]
            raise  # Re-raise the error as this is critical

        try:
            default_session = self.create_session(project_id, "Main Chat")
            project.current_session_id = default_session.id
        except Exception as e_sess:
            logger.error(f"Failed to create default session for project {project_id}: {e_sess}")
            # Clean up, project is in cache but session creation failed
            # Potentially leave project without a current_session_id or handle as critical failure
            project.current_session_id = None  # Ensure it's None if session creation failed

        # Save index now that project (and potentially its current_session_id) is updated
        self._save_projects_index()
        logger.info(f"Created project '{name}' with ID: {project_id}")
        self.projectCreated.emit(project_id)
        return project

    def create_session(self, project_id: str, name: str) -> ChatSession:
        if project_id not in self._projects_cache:  # This check should now pass if called from create_project
            logger.error(f"Attempted to create session for non-existent project ID: {project_id}")
            raise ValueError(f"Project {project_id} not found")

        session_id = str(uuid.uuid4())
        session = ChatSession(id=session_id, project_id=project_id, name=name)
        self._save_session(session)  # Save session to its file
        self._sessions_cache[session_id] = session  # Add to runtime cache
        logger.info(f"Created session '{name}' (ID: {session_id}) in project {project_id}")
        self.sessionCreated.emit(project_id, session_id)
        return session

    def switch_to_project(self, project_id: str) -> bool:
        if project_id not in self._projects_cache:
            logger.error(f"Cannot switch to unknown project: {project_id}")
            return False

        new_project = self._projects_cache[project_id]
        if self._current_project and self._current_project.id == project_id:
            logger.debug(f"Project {project_id} is already current.")
            if new_project.current_session_id and \
                    (not self._current_session or self._current_session.id != new_project.current_session_id):
                self.switch_to_session(new_project.current_session_id)
            elif not new_project.current_session_id:
                sessions = self.get_project_sessions(project_id)
                if sessions: self.switch_to_session(sessions[0].id)
            self.projectSwitched.emit(project_id)
            return True

        self._current_project = new_project
        self._current_session = None

        if self._current_project.current_session_id:
            self.switch_to_session(self._current_project.current_session_id)
        else:
            sessions = self.get_project_sessions(project_id)
            if sessions: self.switch_to_session(sessions[0].id)
            # else: let orchestrator decide if a new default session is needed for an empty project

        logger.info(f"Switched to project: {self._current_project.name}")
        self.projectSwitched.emit(project_id)
        return True

    def switch_to_session(self, session_id: str) -> bool:
        if not self._current_project:
            logger.error("No current project to switch session in")
            return False

        session = self._sessions_cache.get(session_id)
        if not session:
            session = self._load_session(self._current_project.id, session_id)
            if not session:
                logger.error(f"Session {session_id} not found or failed to load for project {self._current_project.id}")
                return False
            self._sessions_cache[session_id] = session

        if session.project_id != self._current_project.id:  # Should not happen if _load_session uses project_id
            logger.error(
                f"Session {session_id} (proj: {session.project_id}) does not belong to current project {self._current_project.id}")
            return False

        self._current_session = session
        if self._current_project.current_session_id != session_id:
            self._current_project.current_session_id = session_id
            self._save_projects_index()

        logger.info(f"Switched to session: {self._current_session.name} in project {self._current_project.name}")
        self.sessionSwitched.emit(self._current_project.id, session_id)
        return True

    def _load_session(self, project_id: str, session_id: str) -> Optional[ChatSession]:
        session_file = os.path.join(self.projects_dir, project_id, "sessions", f"{session_id}.json")
        if not os.path.exists(session_file):
            logger.warning(f"Session file not found: {session_file}")
            return None
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            return ChatSession.from_dict(session_data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load session {session_id} from {session_file}: JSON Error: {e}")
            # Attempt to repair or delete the corrupted file
            try:
                # Backup corrupted file
                backup_file = f"{session_file}.corrupted"
                shutil.copy2(session_file, backup_file)
                logger.warning(f"Backed up corrupted session file to {backup_file}")

                # Optionally, remove the corrupted file to prevent future errors
                os.remove(session_file)
                logger.warning(f"Removed corrupted session file: {session_file}")
            except Exception as backup_error:
                logger.error(f"Failed to backup/remove corrupted session file: {backup_error}")
            return None
        except Exception as e:
            logger.error(f"Failed to load session {session_id} from {session_file}: {e}")
            return None

    def _save_session(self, session: ChatSession):
        if not session.project_id:
            logger.error("Cannot save session without project_id")
            return
        project_sessions_dir = os.path.join(self.projects_dir, session.project_id, "sessions")
        os.makedirs(project_sessions_dir, exist_ok=True)
        session_file = os.path.join(project_sessions_dir, f"{session.id}.json")
        try:
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session {session.id} to {session_file}: {e}")

    def update_current_session_history(self, messages: List[ChatMessage]):  # type: ignore
        if not self._current_session:
            logger.warning("No current session to update")
            return
        self._current_session.message_history = messages  # type: ignore
        self._save_session(self._current_session)

    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        return self._projects_cache.get(project_id)

    def get_session_by_id(self, session_id: str) -> Optional[ChatSession]:
        if session_id in self._sessions_cache:
            return self._sessions_cache[session_id]
        # Attempt to load if not in cache (requires knowing its project_id)
        # This simplified version might not find sessions not already in cache or not belonging to current project easily.
        # For sessions explicitly requested by ID, iterating projects might be needed if project context isn't given.
        if self._current_project:  # Try current project first
            session = self._load_session(self._current_project.id, session_id)
            if session: self._sessions_cache[session_id] = session; return session

        for pid_iter in self._projects_cache:  # Fallback: check all projects
            session = self._load_session(pid_iter, session_id)
            if session: self._sessions_cache[session_id] = session; return session
        return None

    def get_current_project(self) -> Optional[Project]:
        return self._current_project

    def get_current_session(self) -> Optional[ChatSession]:
        return self._current_session

    def get_all_projects(self) -> List[Project]:
        return list(self._projects_cache.values())

    def get_project_sessions(self, project_id: str) -> List[ChatSession]:  # type: ignore
        if project_id not in self._projects_cache: return []
        sessions_dir = os.path.join(self.projects_dir, project_id, "sessions")
        if not os.path.exists(sessions_dir): return []
        sessions = []
        # Track session IDs we've already added
        session_ids_seen = set()

        for filename in os.listdir(sessions_dir):
            if filename.endswith('.json'):
                session_id = filename[:-5]

                # Skip if we've already processed this session ID
                if session_id in session_ids_seen:
                    continue
                session_ids_seen.add(session_id)

                session = self._sessions_cache.get(session_id)
                if not session:
                    loaded_s = self._load_session(project_id, session_id)
                    if loaded_s:
                        self._sessions_cache[session_id] = loaded_s
                        session = loaded_s
                if session:
                    sessions.append(session)

        sessions.sort(key=lambda s: s.created_at if s.created_at else "")
        return sessions

    def get_project_files_dir(self, project_id: Optional[str] = None) -> str:
        target_pid = project_id or (self._current_project.id if self._current_project else None)
        if not target_pid:
            logger.warning("get_project_files_dir called with no project context, returning generic dir.")
            return os.path.join(self.projects_dir, "default_generated_files")
        return os.path.join(self.projects_dir, target_pid, "generated_files")

    def delete_project(self, project_id: str) -> bool:
        if project_id not in self._projects_cache: return False
        try:
            sessions = self.get_project_sessions(project_id)
            for session in sessions: self.delete_session(project_id, session.id, emit_project_deleted=False)
            project_path = os.path.join(self.projects_dir, project_id)
            if os.path.exists(project_path):
                # import shutil # For robust deletion of non-empty dirs
                # shutil.rmtree(project_path) # This is safer for non-empty directories
                # For now, simple rmdir attempts:
                try:
                    files_dir = os.path.join(project_path, "generated_files")
                    sessions_dir = os.path.join(project_path, "sessions")
                    if os.path.exists(files_dir) and not os.listdir(files_dir): os.rmdir(files_dir)
                    if os.path.exists(sessions_dir) and not os.listdir(sessions_dir): os.rmdir(sessions_dir)
                    os.rmdir(project_path)  # Fails if not empty
                except OSError as e_rmdir:
                    logger.warning(
                        f"Could not fully remove project directory {project_path}: {e_rmdir}. May need manual cleanup or shutil.rmtree.")

            del self._projects_cache[project_id]
            self._save_projects_index()
            if self._current_project and self._current_project.id == project_id:
                self._current_project, self._current_session = None, None
            self.projectDeleted.emit(project_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}");
            return False

    def delete_session(self, project_id: str, session_id: str,
                       emit_project_deleted: bool = True):  # Parameter 'emit_project_deleted' seems unused
        session_file = os.path.join(self.projects_dir, project_id, "sessions", f"{session_id}.json")
        if os.path.exists(session_file):
            try:
                os.remove(session_file)
                if session_id in self._sessions_cache: del self._sessions_cache[session_id]
                project = self.get_project_by_id(project_id)
                if project and project.current_session_id == session_id:
                    project.current_session_id = None;
                    self._save_projects_index()
                    if self._current_session and self._current_session.id == session_id: self._current_session = None
                return True
            except Exception as e:
                logger.error(f"Error deleting session file {session_file}: {e}");
                return False
        return False
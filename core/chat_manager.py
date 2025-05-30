# core/chat_manager.py - Enhanced with project iteration and robust file handling
import asyncio
import logging
import os
import uuid
import re
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime # Ensure datetime is imported

from core.code_output_processor import CodeOutputProcessor, CodeQualityLevel
from PySide6.QtCore import QObject, Slot

# Move the TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from core.application_orchestrator import ApplicationOrchestrator

try:
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from core.message_enums import MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from services.project_service import ProjectManager
    from services.upload_service import UploadService
    from core.rag_handler import RagHandler
    from utils import constants # Ensure constants is imported here
    from config import get_gemini_api_key, get_openai_api_key # type: ignore
    from core.user_input_handler import UserInputHandler, UserInputIntent
    from core.plan_and_code_coordinator import PlanAndCodeCoordinator
except ImportError as e:
    ProjectManager = type("ProjectManager", (object,), {}) # type: ignore
    UploadService = type("UploadService", (object,), {}) # type: ignore
    RagHandler = type("RagHandler", (object,), {}) # type: ignore
    logging.getLogger(__name__).critical(f"Critical import error in ChatManager: {e}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class ChatManager(QObject):
    def __init__(self, orchestrator: 'ApplicationOrchestrator', parent: Optional[QObject] = None):
        super().__init__(parent)
        logger.info("ChatManager initializing with project iteration support...")

        self._orchestrator: 'ApplicationOrchestrator' = orchestrator
        self._event_bus: EventBus = self._orchestrator.get_event_bus()
        self._backend_coordinator: BackendCoordinator = self._orchestrator.get_backend_coordinator()
        self._llm_comm_logger: Optional[LlmCommunicationLogger] = self._orchestrator.get_llm_communication_logger()
        self._project_manager: ProjectManager = self._orchestrator.get_project_manager()

        # Optimized code processor
        self._code_processor = CodeOutputProcessor()

        self._upload_service: Optional[UploadService] = self._orchestrator.get_upload_service()
        self._rag_handler: Optional[RagHandler] = self._orchestrator.get_rag_handler()

        # Enhanced code streaming state with performance tracking
        self._active_code_streams: Dict[str, Dict[str, Any]] = {}
        # Structure: {request_id: {'block_id': str, 'is_streaming_code': bool, 'code_fence_count': int, 'start_time': float}}

        if not isinstance(self._project_manager, ProjectManager): # type: ignore
            logger.critical("ChatManager received an invalid ProjectManager instance.")
            raise TypeError("ChatManager requires a valid ProjectManager instance from Orchestrator.")

        self._user_input_handler = UserInputHandler()
        self._plan_and_code_coordinator = PlanAndCodeCoordinator(
            backend_coordinator=self._backend_coordinator,
            event_bus=self._event_bus,
            llm_comm_logger=self._llm_comm_logger,
            parent=self
        )

        self._current_chat_history: List[ChatMessage] = []
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None

        # Optimized backend configuration
        self._active_chat_backend_id: str = constants.DEFAULT_CHAT_BACKEND_ID
        if self._active_chat_backend_id == "ollama_chat_default":
            self._active_chat_model_name: str = constants.DEFAULT_OLLAMA_CHAT_MODEL
        elif self._active_chat_backend_id == "gpt_chat_default":
            self._active_chat_model_name: str = constants.DEFAULT_GPT_CHAT_MODEL
        else:
            self._active_chat_backend_id = "gemini_chat_default" # Fallback to Gemini if not specified
            self._active_chat_model_name: str = constants.DEFAULT_GEMINI_CHAT_MODEL

        self._active_chat_personality_prompt: Optional[
            str] = "You are Ava, a bubbly, enthusiastic, and incredibly helpful AI assistant!"
        self._active_specialized_backend_id: str = constants.GENERATOR_BACKEND_ID
        self._active_specialized_model_name: str = constants.DEFAULT_OLLAMA_GENERATOR_MODEL

        self._active_temperature: float = 0.7
        self._is_chat_backend_configured: bool = False
        self._is_specialized_backend_configured: bool = False
        self._is_rag_ready: bool = False
        self._current_llm_request_id: Optional[str] = None
        self._file_creation_request_ids: Dict[str, str] = {}
        self._llm_terminal_opened: bool = False

        # Performance tracking
        self._response_times: Dict[str, float] = {}
        self._chunk_counts: Dict[str, int] = {}

        # Enhanced streaming with better performance
        self._event_bus.llmStreamChunkReceived.connect(self._handle_llm_stream_chunk)

        self._connect_event_bus_subscriptions()
        logger.info("ChatManager initialized with project iteration support.")

    def initialize(self):
        logger.info("ChatManager optimized initialization...")
        from PySide6.QtCore import QTimer

        async def configure_core_backends_async():
            logger.info("ChatManager: Parallel backend configuration...")

            # Optimized backend configuration - prioritize active backends
            priority_backends = {
                self._active_chat_backend_id: (self._active_chat_model_name, self._active_chat_personality_prompt),
                constants.GENERATOR_BACKEND_ID: (constants.DEFAULT_OLLAMA_GENERATOR_MODEL,
                                                 constants.CODER_AI_SYSTEM_PROMPT)
            }

            # Secondary backends for dropdown availability
            secondary_backends = {
                "gemini_chat_default": (constants.DEFAULT_GEMINI_CHAT_MODEL, self._active_chat_personality_prompt),
                "ollama_chat_default": (constants.DEFAULT_OLLAMA_CHAT_MODEL, self._active_chat_personality_prompt),
                "gpt_chat_default": (constants.DEFAULT_GPT_CHAT_MODEL, self._active_chat_personality_prompt),
            }

            # Configure priority backends first (parallel)
            priority_tasks = []
            for backend_id, (model_name, system_prompt) in priority_backends.items():
                logger.info(f"CM: Priority configuration for '{backend_id}' with model '{model_name}'")
                task = asyncio.create_task(self._configure_backend_async(backend_id, model_name, system_prompt))
                priority_tasks.append(task)

            # Wait for priority backends
            await asyncio.gather(*priority_tasks, return_exceptions=True)

            # Brief pause before secondary backends
            await asyncio.sleep(0.3)

            # Configure secondary backends (parallel, lower priority)
            secondary_tasks = []
            for backend_id, (model_name, system_prompt) in secondary_backends.items():
                if backend_id not in priority_backends:  # Don't reconfigure
                    logger.info(f"CM: Secondary configuration for '{backend_id}'")
                    task = asyncio.create_task(self._configure_backend_async(backend_id, model_name, system_prompt))
                    secondary_tasks.append(task)

            # Wait for secondary backends (non-blocking for UI)
            await asyncio.gather(*secondary_tasks, return_exceptions=True)

            self._check_rag_readiness_and_emit_status()
            logger.info("ChatManager: Optimized backend configuration complete")

        def start_backend_configuration():
            try:
                asyncio.create_task(configure_core_backends_async())
            except Exception as e:
                logger.error(f"Failed to start optimized backend configuration: {e}")
                self._check_rag_readiness_and_emit_status()

        # Faster startup - 1 second delay instead of 2
        QTimer.singleShot(1000, start_backend_configuration)

    async def _configure_backend_async(self, backend_id: str, model_name: str, system_prompt: Optional[str]):
        """Async backend configuration to prevent blocking."""
        try:
            self._configure_backend(backend_id, model_name, system_prompt)
            await asyncio.sleep(0.1)  # Small yield
        except Exception as e:
            logger.error(f"Async backend config error for {backend_id}: {e}")

    def set_active_session(self, project_id: str, session_id: str, history: List[ChatMessage]):
        logger.info(f"CM: Setting active session to P:{project_id}/S:{session_id}. History items: {len(history)}")
        old_project_id = self._current_project_id
        self._current_project_id = project_id
        self._current_session_id = session_id
        self._current_chat_history = list(history)

        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._event_bus.hideLoader.emit()

        self._file_creation_request_ids.clear()
        self._cleanup_code_streams()

        self._event_bus.activeSessionHistoryCleared.emit(project_id, session_id)
        self._event_bus.activeSessionHistoryLoaded.emit(project_id, session_id, self._current_chat_history)
        self._emit_status_update(f"Switched to session.", "#98c379", True, 2000)
        self._check_rag_readiness_and_emit_status()

    def _cleanup_code_streams(self):
        """Optimized cleanup of code streams."""
        if self._active_code_streams:
            logger.info(f"Cleaning up {len(self._active_code_streams)} active code streams")
            for request_id, stream_state in self._active_code_streams.items():
                if stream_state.get('is_streaming_code') and stream_state.get('block_id') and self._llm_comm_logger:
                    self._llm_comm_logger.finish_streaming_code_block(stream_state['block_id'])
            self._active_code_streams.clear()

    def _connect_event_bus_subscriptions(self):
        logger.debug("ChatManager connecting EventBus subscriptions...")
        self._event_bus.userMessageSubmitted.connect(self.handle_user_message)
        self._event_bus.newChatRequested.connect(self.request_new_chat_session)
        self._event_bus.chatLlmPersonalitySubmitted.connect(self._handle_personality_change_event)
        self._event_bus.chatLlmSelectionChanged.connect(self._handle_chat_llm_selection_event)
        self._event_bus.specializedLlmSelectionChanged.connect(self._handle_specialized_llm_selection_event)
        self._event_bus.llmRequestSent.connect(self._handle_llm_request_sent)
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_response_completed)
        self._event_bus.llmResponseError.connect(self._handle_llm_response_error)
        self._event_bus.backendConfigurationChanged.connect(self._handle_backend_reconfigured_event)
        self._event_bus.requestRagScanDirectory.connect(self.request_global_rag_scan_directory)
        self._event_bus.requestProjectFilesUpload.connect(self.handle_project_files_upload_request)
        logger.debug("ChatManager EventBus subscriptions connected.")

    @Slot(str, str)
    def _handle_llm_stream_chunk(self, request_id: str, chunk: str):
        """Optimized streaming chunk handler with performance tracking."""
        import time

        if request_id not in self._active_code_streams:
            # Initialize streaming state
            self._active_code_streams[request_id] = {
                'block_id': None,
                'is_streaming_code': False,
                'code_fence_count': 0,
                'language_hint': 'python',
                'accumulated_chunk': '',
                'start_time': time.time()
            }
            self._chunk_counts[request_id] = 0

        stream_state = self._active_code_streams[request_id]
        self._chunk_counts[request_id] += 1

        # Fast path for non-code chunks
        if not stream_state['is_streaming_code'] and '```' not in chunk:
            return

        # Accumulate chunks for fence detection
        stream_state['accumulated_chunk'] += chunk

        # Optimized fence detection
        fence_matches = list(re.finditer(r'```(\w+)?', stream_state['accumulated_chunk']))

        if fence_matches:
            for match in fence_matches:
                stream_state['code_fence_count'] += 1

                if stream_state['code_fence_count'] % 2 == 1:  # Opening fence
                    language = match.group(1) or 'python'
                    stream_state['language_hint'] = language
                    stream_state['is_streaming_code'] = True

                    if self._llm_comm_logger:
                        block_id = self._llm_comm_logger.start_streaming_code_block(language)
                        stream_state['block_id'] = block_id

                    logger.debug(f"Started code streaming for {request_id} ({language})")

                else:  # Closing fence
                    if stream_state['block_id'] and self._llm_comm_logger:
                        self._llm_comm_logger.finish_streaming_code_block(stream_state['block_id'])

                    # Performance logging
                    elapsed = time.time() - stream_state['start_time']
                    chunk_count = self._chunk_counts.get(request_id, 0)
                    logger.debug(f"Finished code streaming for {request_id}: {chunk_count} chunks in {elapsed:.2f}s")

                    stream_state['is_streaming_code'] = False
                    stream_state['block_id'] = None

            stream_state['accumulated_chunk'] = ''

        # Stream clean chunks to logger
        elif stream_state['is_streaming_code'] and stream_state['block_id'] and self._llm_comm_logger:
            clean_chunk = re.sub(r'```\w*', '', chunk)
            if clean_chunk:
                self._llm_comm_logger.stream_code_chunk(stream_state['block_id'], clean_chunk)

    def _emit_status_update(self, message: str, color: str, is_temporary: bool = False, duration_ms: int = 0):
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, is_temporary, duration_ms)

    def _log_llm_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            log_prefix = f"[P:{self._current_project_id[:6]}/S:{self._current_session_id[:6]}]" if self._current_project_id and self._current_session_id else "[NoActiveSession]"
            self._llm_comm_logger.log_message(f"{log_prefix} {sender}", message)
            if not self._llm_terminal_opened:
                self._llm_terminal_opened = True
                logger.info("ChatManager: Auto-opening LLM terminal for first communication")
                self._event_bus.showLlmLogWindowRequested.emit()
        else:
            logger.info(f"LLM_COMM_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def _configure_backend(self, backend_id: str, model_name: str, system_prompt: Optional[str]):
        logger.info(f"CM: Configuring backend '{backend_id}' with model '{model_name}'")
        try:
            api_key_to_use: Optional[str] = None

            if backend_id == "gemini_chat_default":
                api_key_to_use = get_gemini_api_key()
            elif backend_id == "gpt_chat_default":
                api_key_to_use = get_openai_api_key()
            elif backend_id in [constants.GENERATOR_BACKEND_ID, "ollama_chat_default"]:
                api_key_to_use = None

            if backend_id in ["gemini_chat_default", "gpt_chat_default"] and not api_key_to_use:
                err_msg = f"{backend_id.split('_').upper()} API Key not found. Set in .env"
                logger.error(err_msg)
                if backend_id == self._active_chat_backend_id:
                    self._is_chat_backend_configured = False
                elif backend_id == self._active_specialized_backend_id:
                    self._is_specialized_backend_configured = False
                self._event_bus.backendConfigurationChanged.emit(backend_id, model_name, False, [])
                self._emit_status_update(err_msg, "#FF6B6B")
                return

            self._backend_coordinator.configure_backend(backend_id=backend_id, api_key=api_key_to_use,
                                                        model_name=model_name, system_prompt=system_prompt)
        except Exception as e:
            logger.error(f"CM: Error configuring backend '{backend_id}': {e}", exc_info=True)
            self._emit_status_update(f"Failed to configure {backend_id}: {e}", "#FF6B6B")

    def _detect_file_creation_intent(self, user_text: str) -> Optional[str]:
        """Optimized file creation detection with common patterns."""
        # Pre-compiled patterns for better performance
        patterns = [
            r"create (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"create ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"make (?:a )?file ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"write (?:a )?file called ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"generate ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
            r"save (?:this )?as ['\"]?([a-zA-Z0-9_\-./]+\.py)['\"]?",
        ]

        for pattern in patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                filename = match.group(1)
                logger.debug(f"Detected file creation intent for: {filename}")
                return filename
        return None

    def _detect_task_type(self, user_text: str, filename: str) -> str:
        """Fast task type detection using keyword matching."""
        user_lower = user_text.lower()
        filename_lower = filename.lower()

        # Quick filename-based detection
        filename_indicators = {
            'api': ['api', 'server', 'endpoint', 'route'],
            'data_processing': ['data', 'process', 'parse', 'etl'],
            'ui': ['ui', 'window', 'dialog', 'widget', 'gui'],
            'utility': ['util', 'helper', 'tool', 'lib']
        }

        for task_type, keywords in filename_indicators.items():
            if any(kw in filename_lower for kw in keywords):
                return task_type

        # Content-based detection (optimized patterns)
        if any(term in user_lower for term in ['api', 'endpoint', 'fastapi', 'flask', 'rest']):
            return 'api'
        elif any(term in user_lower for term in ['data', 'csv', 'pandas', 'process', 'analyze']):
            return 'data_processing'
        elif any(term in user_lower for term in ['ui', 'interface', 'pyside', 'window', 'dialog']):
            return 'ui'

        return 'general'

    def _get_specialized_prompt(self, task_type: str, user_text: str, filename: str) -> str:
        """Optimized prompt generation with cached templates."""
        base_instruction = f"You are Devstral, an expert coding assistant. Create a file called '{filename}' based on this request:\n\n{user_text}\n\n"

        prompt_map = {
            'api': constants.API_DEVELOPMENT_PROMPT,
            'data_processing': constants.DATA_PROCESSING_PROMPT,
            'ui': constants.UI_DEVELOPMENT_PROMPT,
            'utility': constants.UTILITY_DEVELOPMENT_PROMPT,
            'general': constants.GENERAL_CODING_PROMPT
        }

        specific_prompt = prompt_map.get(task_type, constants.GENERAL_CODING_PROMPT)
        return base_instruction + specific_prompt + f"\n\nCreate complete, production-ready code for '{filename}' that fulfills the user's request."

    def _create_file_creation_prompt(self, user_text: str, filename: str) -> str:
        task_type = self._detect_task_type(user_text, filename)
        logger.debug(f"CM: Detected task type '{task_type}' for file '{filename}'")
        return self._get_specialized_prompt(task_type, user_text, filename)

    # ✨ NEW: Project iteration support methods
    def _create_project_iteration_prompt(self, user_text: str, project_context: Optional[str] = None) -> str:
        """Create a specialized prompt for project iteration requests"""
        base_prompt = f"""You are an expert software architect and developer helping to improve an existing codebase.

PROJECT ITERATION REQUEST:
{user_text}

CURRENT PROJECT CONTEXT:
{project_context or "No specific project context available - use RAG context below if provided."}

CRITICAL INSTRUCTIONS:
1. Analyze the existing code structure and patterns
2. Provide specific, actionable improvements
3. Maintain backward compatibility unless explicitly asked to break it
4. Follow the existing code style and conventions
5. Focus on the specific improvements requested
6. If creating new files, output complete code in fenced blocks
7. If modifying existing files, clearly indicate what changes to make

RESPONSE FORMAT:
- Provide analysis of current state
- Suggest specific improvements
- If creating files, use: ```python\\n[CODE]\\n```
- If modifying files, clearly indicate changes needed

Please analyze the request and provide thoughtful improvements to enhance the existing codebase."""

        return base_prompt

    def _get_project_context(self) -> Optional[str]:
        """Get context about the current project for iteration requests"""
        if not self._current_project_id:
            return None

        project = self._project_manager.get_project_by_id(self._current_project_id)
        if not project:
            return None

        context_parts = [
            f"Project: {project.name}",
            f"Description: {project.description or 'No description available'}"
        ]

        # Add recent chat history context (last 3 messages for context)
        if len(self._current_chat_history) > 0:
            recent_messages = self._current_chat_history[-3:]
            context_parts.append("\nRecent Conversation Context:")
            for msg in recent_messages:
                if msg.role in [USER_ROLE, MODEL_ROLE]:
                    content_preview = msg.text[:200] + "..." if len(msg.text) > 200 else msg.text
                    context_parts.append(f"- {msg.role.title()}: {content_preview}")

        return "\n".join(context_parts)

    @Slot(str, str, bool, list)
    def _handle_backend_reconfigured_event(self, backend_id: str, model_name: str, is_configured: bool,
                                           available_models: list):
        is_active_chat = (backend_id == self._active_chat_backend_id and model_name == self._active_chat_model_name)
        is_active_spec = (
                backend_id == self._active_specialized_backend_id and model_name == self._active_specialized_model_name)

        if backend_id == self._active_chat_backend_id:
            self._is_chat_backend_configured = is_configured
        elif backend_id == self._active_specialized_backend_id:
            self._is_specialized_backend_configured = is_configured

        if is_active_chat:
            status_msg = f"Ready. Using {self._active_chat_model_name}" if is_configured else f"Failed to config Chat LLM: {self._backend_coordinator.get_last_error_for_backend(backend_id) or 'Unknown'}"
            status_color = "#98c379" if is_configured else "#FF6B6B"
            self._emit_status_update(status_msg, status_color, not is_configured)
        elif is_active_spec:
            status_msg = f"Specialized LLM {model_name} ready." if is_configured else f"Specialized LLM Error: {self._backend_coordinator.get_last_error_for_backend(backend_id) or 'Unknown'}"
            status_color = "#98c379" if is_configured else "#FF6B6B"
            self._emit_status_update(status_msg, status_color, True, 5000 if not is_configured else 3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(str, list)
    def handle_user_message(self, text: str, image_data: List[Dict[str, Any]]):
        """✨ ENHANCED: User message handling with project iteration support."""
        if not self._current_project_id or not self._current_session_id:
            self._emit_status_update("Error: No active project or session.", "#FF6B6B", True, 3000)
            return

        logger.info(
            f"CM: User message for P:{self._current_project_id}/S:{self._current_session_id} - '{text[:50]}...'")

        # Fast intent processing
        processed_input = self._user_input_handler.process_input(text, image_data)
        user_msg_txt = processed_input.original_query.strip()

        if not user_msg_txt and not (image_data and processed_input.intent == UserInputIntent.NORMAL_CHAT):
            return

        # Quick busy check
        if self._current_llm_request_id and processed_input.intent in [UserInputIntent.NORMAL_CHAT,
                                                                       UserInputIntent.FILE_CREATION_REQUEST,
                                                                       UserInputIntent.PROJECT_ITERATION_REQUEST]:
            self._emit_status_update("Please wait for the current AI response.", "#e5c07b", True, 2500)
            return

        # Add user message to history
        user_msg_parts = [user_msg_txt] if user_msg_txt else []
        if image_data:
            user_msg_parts.extend(image_data) # type: ignore

        user_message = ChatMessage(role=USER_ROLE, parts=user_msg_parts)
        self._current_chat_history.append(user_message)
        self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, user_message)
        self._log_llm_comm("USER", user_msg_txt)

        # ✨ ENHANCED: Route to appropriate handler with new iteration support
        if processed_input.intent == UserInputIntent.NORMAL_CHAT:
            asyncio.create_task(self._handle_normal_chat_async(user_msg_txt, image_data))
        elif processed_input.intent == UserInputIntent.FILE_CREATION_REQUEST:
            self._handle_file_creation_request(user_msg_txt, processed_input.data.get('filename'))
        elif processed_input.intent == UserInputIntent.PLAN_THEN_CODE_REQUEST:
            self._handle_plan_then_code_request(user_msg_txt)
        elif processed_input.intent == UserInputIntent.PROJECT_ITERATION_REQUEST:
            # ✨ NEW: Handle project iteration requests
            asyncio.create_task(self._handle_project_iteration_request(user_msg_txt, image_data))
        else:
            unknown_intent_msg = ChatMessage(role=ERROR_ROLE,
                                             parts=[f"[System: Unknown request type: {user_msg_txt[:50]}...]"])
            self._current_chat_history.append(unknown_intent_msg)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              unknown_intent_msg)

    async def _handle_project_iteration_request(self, user_msg_txt: str, image_data: List[Dict[str, Any]]):
        """✨ NEW: Handle requests to iterate/improve existing project code"""
        self._event_bus.showLoader.emit("Analyzing project for improvements...")

        if not self._is_chat_backend_configured:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=["[Error: Chat Backend not ready.]"])
            self._current_chat_history.append(err_msg_obj)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update("Cannot iterate: Chat AI backend not configured.", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        # Get project context
        project_context = self._get_project_context()

        # Create specialized iteration prompt
        iteration_prompt = self._create_project_iteration_prompt(user_msg_txt, project_context)

        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[iteration_prompt])]

        # Enhanced RAG for project iteration - we want MORE context for improvements
        should_use_rag = (self._rag_handler and self._is_rag_ready and
                          self._rag_handler.should_perform_rag(user_msg_txt, self._is_rag_ready, self._is_rag_ready))

        if should_use_rag and self._rag_handler:
            logger.info(f"CM: Performing enhanced RAG for iteration: '{user_msg_txt[:50]}'")
            query_entities = self._rag_handler.extract_code_entities(user_msg_txt)

            # Get MORE context for iteration requests (modification_request=True)
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                query=user_msg_txt,
                query_entities=query_entities,
                project_id=self._current_project_id,
                explicit_focus_paths=[],
                implicit_focus_paths=[],
                is_modification_request=True  # ✨ This gets more context for iteration
            )

            if rag_context_str:
                logger.info(f"CM: Enhanced RAG context found from collections: {queried_collections}")
                rag_system_message = ChatMessage(role=SYSTEM_ROLE, parts=[rag_context_str],
                                                 metadata={"is_rag_context": True,
                                                           "queried_collections": queried_collections,
                                                           "is_iteration_context": True})
                history_for_llm.insert(-1, rag_system_message) # type: ignore

                rag_notification_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE, parts=[
                    f"[System: Enhanced project analysis with context from: {', '.join(queried_collections)}.]"],
                                                   metadata={"is_internal": True})
                self._current_chat_history.append(rag_notification_msg)
                if self._current_project_id and self._current_session_id:
                    self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              rag_notification_msg)
                self._log_llm_comm("ITERATION_RAG",
                                   rag_context_str[:200] + "..." if len(rag_context_str) > 200 else rag_context_str)

        # Start LLM request with higher temperature for creativity in improvements
        iteration_temperature = min(self._active_temperature + 0.2, 1.5)  # Slightly more creative

        self._event_bus.updateLoaderMessage.emit(f"Analyzing with {self._active_chat_model_name}...")
        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                   history_for_llm, {
                                                                                       "temperature": iteration_temperature})
        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[f"[Error in iteration: {err}]"])
            self._current_chat_history.append(err_msg_obj)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start iteration analysis: {err or 'Unknown'}", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        # Track request timing
        import time
        self._response_times[req_id] = time.time()

        self._current_llm_request_id = req_id
        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(placeholder)
        if self._current_project_id and self._current_session_id:
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, placeholder)

        self._backend_coordinator.start_llm_streaming_task(req_id, self._active_chat_backend_id, history_for_llm, False,
                                                           {"temperature": iteration_temperature},
                                                           {"purpose": "project_iteration",
                                                            "project_id": self._current_project_id,
                                                            "session_id": self._current_session_id})

    async def _handle_normal_chat_async(self, user_msg_txt: str, image_data: List[Dict[str, Any]]):
        """Optimized normal chat handling with faster RAG processing."""
        self._event_bus.showLoader.emit("Thinking...")

        if not self._is_chat_backend_configured:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=["[Error: Chat Backend not ready.]"])
            self._current_chat_history.append(err_msg_obj)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update("Cannot send: Chat AI backend not configured.", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        history_for_llm = self._current_chat_history[:]

        # Optimized RAG decision
        should_use_rag = (self._rag_handler and self._is_rag_ready and
                          self._rag_handler.should_perform_rag(user_msg_txt, self._is_rag_ready, self._is_rag_ready))

        if should_use_rag and self._upload_service:
            self._event_bus.updateLoaderMessage.emit("Searching context...")

            # Fast RAG readiness check
            if not self._upload_service._embedder_ready: # type: ignore
                self._emit_status_update("Waiting for RAG system...", "#e5c07b", False)
                embedder_ready = await self._upload_service.wait_for_embedder_ready( # type: ignore
                    timeout_seconds=5.0)  # Reduced timeout
                if not embedder_ready:
                    self._emit_status_update("RAG system not ready, continuing...", "#e5c07b", True, 2000)
                    should_use_rag = False

        if should_use_rag and self._rag_handler:
            logger.info(f"CM: Performing optimized RAG for query: '{user_msg_txt[:50]}'")
            query_entities = self._rag_handler.extract_code_entities(user_msg_txt)
            rag_context_str, queried_collections = self._rag_handler.get_formatted_context(
                query=user_msg_txt, query_entities=query_entities, project_id=self._current_project_id,
                explicit_focus_paths=[], implicit_focus_paths=[], is_modification_request=False
            )

            if rag_context_str:
                logger.info(f"CM: RAG context found from collections: {queried_collections}")
                rag_system_message = ChatMessage(role=SYSTEM_ROLE, parts=[rag_context_str],
                                                 metadata={"is_rag_context": True,
                                                           "queried_collections": queried_collections})
                history_for_llm.insert(-1, rag_system_message) # type: ignore

                rag_notification_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE, parts=[
                    f"[System: RAG used. Context from: {', '.join(queried_collections)}.]"],
                                                   metadata={"is_internal": True})
                self._current_chat_history.append(rag_notification_msg)
                if self._current_project_id and self._current_session_id:
                    self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                              rag_notification_msg)
                self._log_llm_comm("RAG Context",
                                   rag_context_str[:200] + "..." if len(rag_context_str) > 200 else rag_context_str)

        # Start LLM request
        self._event_bus.updateLoaderMessage.emit(f"Sending to {self._active_chat_model_name}...")
        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                   history_for_llm, {
                                                                                       "temperature": self._active_temperature})
        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[f"[Error sending: {err}]"])
            self._current_chat_history.append(err_msg_obj)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start chat: {err or 'Unknown'}", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        # Track request timing
        import time
        self._response_times[req_id] = time.time()

        self._current_llm_request_id = req_id
        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(placeholder)
        if self._current_project_id and self._current_session_id:
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, placeholder)

        self._backend_coordinator.start_llm_streaming_task(req_id, self._active_chat_backend_id, history_for_llm, False,
                                                           {"temperature": self._active_temperature},
                                                           {"purpose": "normal_chat",
                                                            "project_id": self._current_project_id,
                                                            "session_id": self._current_session_id})

    def _handle_plan_then_code_request(self, user_msg_txt: str):
        """Enhanced plan-then-code with better error handling."""
        if not self._is_chat_backend_configured or not self._is_specialized_backend_configured:
            self._emit_status_update("Both Chat and Specialized LLMs must be configured for autonomous coding.",
                                     "#e06c75", True, 5000)
            return

        current_project_dir = self._get_current_project_directory()
        task_type = self._detect_task_type(user_msg_txt, "multi_file_project")

        self._log_llm_comm("AUTONOMOUS_CODING_REQUEST", f"Starting autonomous coding for: {user_msg_txt[:100]}...")

        if hasattr(self._plan_and_code_coordinator, 'is_busy') and self._plan_and_code_coordinator.is_busy():
            self._emit_status_update("Autonomous coding already in progress. Please wait.", "#e5c07b", True, 3000)
            return

        try:
            success = self._plan_and_code_coordinator.start_autonomous_coding(
                user_query=user_msg_txt,
                planner_backend=self._active_chat_backend_id,
                planner_model=self._active_chat_model_name,
                coder_backend=self._active_specialized_backend_id,
                coder_model=self._active_specialized_model_name,
                project_dir=current_project_dir,
                project_id=self._current_project_id,
                session_id=self._current_session_id,
                task_type=task_type
            )

            if not success:
                self._emit_status_update("Failed to start autonomous coding sequence", "#e06c75", True, 3000)
                self._log_llm_comm("AUTONOMOUS_CODING_ERROR", "Failed to start sequence")

        except Exception as e:
            logger.error(f"Error starting autonomous coding: {e}", exc_info=True)
            self._emit_status_update(f"Autonomous coding error: {str(e)}", "#e06c75", True, 5000)
            self._log_llm_comm("AUTONOMOUS_CODING_EXCEPTION", f"Exception: {e}")

    def _get_current_project_directory(self) -> str:
        """Optimized project directory resolution, ensuring it's in USER_DATA_DIR."""
        # import os # Already imported at top
        # from datetime import datetime # Already imported at top
        # from utils import constants # Already imported at top

        # Priority 1: If a current project is active, use its specific 'generated_files' directory
        if self._current_project_id and self._project_manager:
            project_obj = self._project_manager.get_project_by_id(self._current_project_id)
            if project_obj:
                # This correctly uses the project manager's isolated directory structure within USER_DATA_DIR
                project_output_dir = self._project_manager.get_project_files_dir(self._current_project_id)
                os.makedirs(project_output_dir, exist_ok=True)
                logger.info(f"CM: Using active project directory for generation: {project_output_dir}")
                return project_output_dir

        # Priority 2: Fallback for when no specific project context is found
        # Ensure this fallback is OUTSIDE the application's source code directory and INSIDE USER_DATA_DIR.
        base_output_dir_for_no_context = os.path.join(constants.USER_DATA_DIR, "ava_generated_projects_no_context")
        # Create a timestamped or default directory within this "no context" base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_output_dir = os.path.join(base_output_dir_for_no_context, f"untitled_project_{timestamp}")

        os.makedirs(project_output_dir, exist_ok=True)
        logger.info(f"CM: Using fallback project directory (no active project): {project_output_dir}")
        return project_output_dir


    def _extract_code_from_response(self, response_text: str, filename: str = "generated_file.py") -> Optional[str]:
        """Enhanced code extraction using optimized processor."""
        if not response_text or not response_text.strip():
            logger.warning("Empty response text for code extraction")
            return None

        try:
            # Use the optimized code processor
            extracted_code, quality, processing_notes = self._code_processor.process_llm_response(
                response_text, filename, "python"
            )

            # Log extraction results with performance info
            self._log_llm_comm("CODE_EXTRACTION",
                               f"File: {filename}, Quality: {quality.name if quality else 'None'}, "
                               f"Notes: {', '.join(processing_notes)}")

            if extracted_code:
                if quality in [CodeQualityLevel.EXCELLENT, CodeQualityLevel.GOOD, CodeQualityLevel.ACCEPTABLE]:
                    logger.info(f"Successfully extracted code for {filename} (Quality: {quality.name})")
                    return extracted_code
                elif quality == CodeQualityLevel.POOR:
                    logger.warning(f"Poor quality code extracted for {filename}: {', '.join(processing_notes)}")
                    return extracted_code  # Still return it with warnings
                else:
                    logger.error(f"Unusable code extracted for {filename}: {', '.join(processing_notes)}")
                    return None
            else:
                logger.warning(f"No code could be extracted for {filename}: {', '.join(processing_notes)}")
                return None

        except Exception as e:
            logger.error(f"Error in code extraction for {filename}: {e}", exc_info=True)
            self._log_llm_comm("CODE_EXTRACTION_ERROR", f"Error: {e}")
            return self._basic_code_extraction_fallback(response_text)

    def _basic_code_extraction_fallback(self, response_text: str) -> Optional[str]:
        """Fast fallback code extraction."""
        patterns = [
            r"```python\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @Slot()
    def request_new_chat_session(self):
        logger.info("CM: New chat session requested by user/UI.")
        if not self._current_project_id:
            self._emit_status_update("Cannot start new chat: No active project.", "#e06c75", True, 3000)
            return
        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(request_id=self._current_llm_request_id)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._event_bus.hideLoader.emit()
        self._event_bus.createNewSessionForProjectRequested.emit(self._current_project_id)

    @Slot(str)
    def request_global_rag_scan_directory(self, dir_path: str):
        if not self._upload_service or not self._upload_service.is_vector_db_ready(constants.GLOBAL_COLLECTION_ID):
            self._emit_status_update(f"RAG system not ready for Global Knowledge. Cannot scan.", "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting GLOBAL RAG scan for directory: {dir_path}")
        self._event_bus.showLoader.emit(f"Scanning '{os.path.basename(dir_path)}' for Global RAG...")

        rag_message: Optional[ChatMessage] = None
        try:
            rag_message = self._upload_service.process_directory_for_context(dir_path,
                                                                             collection_id=constants.GLOBAL_COLLECTION_ID)
        finally:
            self._event_bus.hideLoader.emit()

        if rag_message and self._current_project_id and self._current_session_id:
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_SCAN_GLOBAL", rag_message.text)
            status_color = "#FF6B6B" if rag_message.role == ERROR_ROLE else "#98c379"
            self._emit_status_update(
                f"Global RAG Scan complete{' with errors' if rag_message.role == ERROR_ROLE else ''}!", status_color,
                True, 3000 if rag_message.role != ERROR_ROLE else 5000)
        elif rag_message:
            logger.info(f"Global RAG Scan completed. Message (not added to active chat): {rag_message.text}")
        else:
            self._emit_status_update(f"Global RAG Scan failed or returned no message.", "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(list, str)
    def handle_project_files_upload_request(self, file_paths: List[str], project_id: str):
        if not project_id:
            logger.error("CM: Project RAG file upload requested without a project_id.")
            self._emit_status_update("Cannot add files to project RAG: Project ID missing.", "#e06c75", True, 3000)
            return

        if not self._upload_service or not self._upload_service.is_vector_db_ready(project_id):
            self._emit_status_update(f"RAG system not ready for project '{project_id[:8]}...'. Cannot add files.",
                                     "#FF6B6B", True, 4000)
            return

        logger.info(f"CM: Requesting Project RAG file upload for {len(file_paths)} files, project: {project_id}")
        self._event_bus.showLoader.emit(f"Adding {len(file_paths)} files to RAG for '{project_id[:8]}...'")

        rag_message: Optional[ChatMessage] = None
        try:
            if self._upload_service: # Check if upload_service is not None
                rag_message = self._upload_service.process_files_for_context(file_paths, collection_id=project_id)
            else:
                logger.error("UploadService is None, cannot process files.")
                rag_message = ChatMessage(role=ERROR_ROLE, parts=["[System Error: Upload service not available]"])

        finally:
            self._event_bus.hideLoader.emit()

        if rag_message and self._current_project_id == project_id and self._current_session_id:
            self._current_chat_history.append(rag_message)
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          rag_message)
            self._log_llm_comm(f"RAG_UPLOAD_P:{project_id[:6]}", rag_message.text)
            status_color = "#FF6B6B" if rag_message.role == ERROR_ROLE else "#98c379"
            self._emit_status_update(
                f"Project RAG file add for '{project_id[:8]}' complete{' with errors' if rag_message.role == ERROR_ROLE else ''}!",
                status_color, True, 3000 if rag_message.role != ERROR_ROLE else 5000)
        elif rag_message:
            logger.info(
                f"Project RAG file add for project '{project_id}' completed. Message (not added to active chat): {rag_message.text}")
            self._emit_status_update(f"Files added to RAG for (non-active) project '{project_id[:8]}'.", "#56b6c2",
                                     True, 4000)
        else:
            self._emit_status_update(f"Project RAG file add for '{project_id[:8]}' failed or returned no message.",
                                     "#FF6B6B", True, 3000)

        self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_llm_request_sent(self, backend_id: str, request_id: str):
        if request_id == self._current_llm_request_id and backend_id == self._active_chat_backend_id:
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

    @Slot(str, object, dict)
    def _handle_llm_response_completed(self, request_id: str, completed_message_obj: object, usage_stats_dict: dict):
        """Enhanced LLM response handling with performance tracking."""
        import time

        meta_pid = usage_stats_dict.get("project_id")
        meta_sid = usage_stats_dict.get("session_id")
        purpose = usage_stats_dict.get("purpose")

        # Performance tracking
        if request_id in self._response_times:
            elapsed = time.time() - self._response_times[request_id]
            chunk_count = self._chunk_counts.get(request_id, 0)
            logger.info(f"CM: Request {request_id} completed in {elapsed:.2f}s with {chunk_count} chunks")
            del self._response_times[request_id]
            self._chunk_counts.pop(request_id, None)

        # Clean up code streaming state
        if request_id in self._active_code_streams:
            stream_state = self._active_code_streams[request_id]
            if stream_state.get('is_streaming_code') and stream_state.get('block_id') and self._llm_comm_logger:
                self._llm_comm_logger.finish_streaming_code_block(stream_state['block_id'])
            del self._active_code_streams[request_id]

        if request_id == self._current_llm_request_id and meta_pid == self._current_project_id and meta_sid == self._current_session_id:
            self._event_bus.hideLoader.emit()

            if isinstance(completed_message_obj, ChatMessage): # type: ignore
                self._log_llm_comm(self._active_chat_backend_id.upper() + " RESPONSE",
                                   completed_message_obj.text[:200] + "..." if len( # type: ignore
                                       completed_message_obj.text) > 200 else completed_message_obj.text) # type: ignore

                # Handle file creation with optimized code extraction
                if purpose in ["file_creation",
                               "enhanced_file_creation"] and request_id in self._file_creation_request_ids:
                    filename = self._file_creation_request_ids.pop(request_id)

                    extracted_code = self._extract_code_from_response(completed_message_obj.text, filename) # type: ignore

                    if extracted_code:
                        logger.info(f"CM: Successfully extracted code for file '{filename}', sending to code viewer")

                        clean_code = self._code_processor.clean_and_format_code(extracted_code)
                        self._event_bus.modificationFileReadyForDisplay.emit(filename, clean_code)

                        file_creation_msg = ChatMessage(
                            id=uuid.uuid4().hex,
                            role=SYSTEM_ROLE,
                            parts=[f"[File created: {filename} - View in Code Viewer ✅]"],
                            metadata={"is_file_creation": True, "filename": filename}
                        )
                        self._current_chat_history.append(file_creation_msg)
                        if self._current_project_id and self._current_session_id:
                             self._event_bus.newMessageAddedToHistory.emit(
                                self._current_project_id, self._current_session_id, file_creation_msg)
                    else:
                        logger.warning(f"CM: Could not extract code from response for file '{filename}'")

                        try:
                            suggestions = self._code_processor.suggest_fixes_for_poor_code(
                                completed_message_obj.text, filename # type: ignore
                            )
                            suggestion_text = f"Code extraction failed for {filename}. Suggestions: {'; '.join(suggestions)}"
                        except:
                            suggestion_text = f"Could not extract code for {filename}. Raw response may contain issues."

                        error_msg = ChatMessage(
                            id=uuid.uuid4().hex,
                            role=ERROR_ROLE,
                            parts=[f"[{suggestion_text}]"],
                            metadata={"is_extraction_error": True, "filename": filename}
                        )
                        self._current_chat_history.append(error_msg)
                        if self._current_project_id and self._current_session_id:
                            self._event_bus.newMessageAddedToHistory.emit(
                                self._current_project_id, self._current_session_id, error_msg)

                # ✨ NEW: Handle project iteration responses
                elif purpose == "project_iteration":
                    # Check if the response contains code that should be displayed in code viewer
                    if "```" in completed_message_obj.text: # type: ignore
                        # Try to extract any code blocks for display
                        code_blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```',
                                                 completed_message_obj.text, re.DOTALL | re.IGNORECASE) # type: ignore

                        for i, code_block in enumerate(code_blocks):
                            if code_block.strip():
                                # Generate a filename for iteration code
                                filename = f"iteration_improvement_{i + 1}.py"
                                clean_code = self._code_processor.clean_and_format_code(code_block.strip())
                                self._event_bus.modificationFileReadyForDisplay.emit(filename, clean_code)

                                # Add notification about code being available
                                code_ready_msg = ChatMessage(
                                    id=uuid.uuid4().hex,
                                    role=SYSTEM_ROLE,
                                    parts=[f"[Improvement code available: {filename} - View in Code Viewer ✨]"],
                                    metadata={"is_iteration_code": True, "filename": filename}
                                )
                                self._current_chat_history.append(code_ready_msg)
                                if self._current_project_id and self._current_session_id:
                                    self._event_bus.newMessageAddedToHistory.emit(
                                        self._current_project_id, self._current_session_id, code_ready_msg)

                    # Always update the message in history for project iteration
                    updated = False
                    for i, msg in enumerate(self._current_chat_history):
                        if msg.id == request_id:
                            self._current_chat_history[i] = completed_message_obj # type: ignore
                            self._current_chat_history[i].loading_state = MessageLoadingState.COMPLETED
                            updated = True
                            break

                    if not updated:
                        self._current_chat_history.append(completed_message_obj) # type: ignore

                # Handle regular responses (normal chat)
                else:
                    # Update the message in history
                    updated = False
                    for i, msg in enumerate(self._current_chat_history):
                        if msg.id == request_id:
                            self._current_chat_history[i] = completed_message_obj # type: ignore
                            self._current_chat_history[i].loading_state = MessageLoadingState.COMPLETED
                            updated = True
                            break

                    if not updated:
                        self._current_chat_history.append(completed_message_obj) # type: ignore

                # Emit finalization signal
                if self._current_project_id and self._current_session_id:
                    self._event_bus.messageFinalizedForSession.emit(
                        self._current_project_id,
                        self._current_session_id,
                        request_id,
                        completed_message_obj,
                        usage_stats_dict,
                        False  # is_error = False
                    )

            # Clean up and update UI
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)

            # ✨ Enhanced status messages
            if purpose == "project_iteration":
                self._emit_status_update(f"Project analysis complete. Using {self._active_chat_model_name}", "#98c379")
            else:
                self._emit_status_update(f"Ready. Last: {self._active_chat_model_name}", "#98c379")

            self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_llm_response_error(self, request_id: str, error_message_str: str):
        # Performance cleanup
        if request_id in self._response_times:
            del self._response_times[request_id]
        self._chunk_counts.pop(request_id, None)

        # Clean up code streaming state
        if request_id in self._active_code_streams:
            stream_state = self._active_code_streams[request_id]
            if stream_state.get('is_streaming_code') and stream_state.get('block_id') and self._llm_comm_logger:
                self._llm_comm_logger.finish_streaming_code_block(stream_state['block_id'])
            del self._active_code_streams[request_id]

        if request_id == self._current_llm_request_id:
            self._event_bus.hideLoader.emit()
            self._log_llm_comm(f"{self._active_chat_backend_id.upper()} ERROR", error_message_str)

            if request_id in self._file_creation_request_ids:
                filename = self._file_creation_request_ids.pop(request_id)
                logger.warning(f"CM: File creation failed for '{filename}': {error_message_str}")

            err_chat_msg = ChatMessage(id=request_id, role=ERROR_ROLE, parts=[f"[AI Error: {error_message_str}]"],
                                       loading_state=MessageLoadingState.ERROR)
            updated = False
            for i, msg in enumerate(self._current_chat_history):
                if msg.id == request_id:
                    self._current_chat_history[i] = err_chat_msg
                    updated = True
                    break
            if not updated:
                self._current_chat_history.append(err_chat_msg)

            if self._current_project_id and self._current_session_id:
                self._event_bus.messageFinalizedForSession.emit(self._current_project_id, self._current_session_id,
                                                                request_id, err_chat_msg, {}, True)
            self._current_llm_request_id = None
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._emit_status_update(f"Error: {error_message_str[:60]}...", "#FF6B6B")
            self._check_rag_readiness_and_emit_status()

    @Slot(str, str)
    def _handle_chat_llm_selection_event(self, backend_id: str, model_name: str):
        if self._active_chat_backend_id != backend_id or self._active_chat_model_name != model_name:
            self._active_chat_backend_id, self._active_chat_model_name = backend_id, model_name
            self._configure_backend(backend_id, model_name, self._active_chat_personality_prompt)

    @Slot(str, str)
    def _handle_specialized_llm_selection_event(self, backend_id: str, model_name: str):
        if self._active_specialized_backend_id != backend_id or self._active_specialized_model_name != model_name:
            self._active_specialized_backend_id, self._active_specialized_model_name = backend_id, model_name
            self._configure_backend(backend_id, model_name, constants.CODER_AI_SYSTEM_PROMPT)

    @Slot(str, str)
    def _handle_personality_change_event(self, new_prompt: str, backend_id_for_persona: str):
        if backend_id_for_persona == self._active_chat_backend_id:
            self._active_chat_personality_prompt = new_prompt.strip() if new_prompt and new_prompt.strip() else None
            self._configure_backend(self._active_chat_backend_id, self._active_chat_model_name,
                                    self._active_chat_personality_prompt)

    def _check_rag_readiness_and_emit_status(self):
        """Optimized RAG status checking."""
        if not self._upload_service or not hasattr(self._upload_service,
                                                   '_vector_db_service') or not self._upload_service._vector_db_service: # type: ignore
            self._is_rag_ready = False
            self._event_bus.ragStatusChanged.emit(False, "RAG Not Ready (Service Error)", "#e06c75")
            return

        rag_status_message = "RAG Status: Initializing..."
        rag_status_color = constants.TIMESTAMP_COLOR_HEX
        current_context_rag_ready = False

        if not self._upload_service._embedder_ready: # type: ignore
            rag_status_message = "RAG: Initializing embedder..."
            rag_status_color = "#e5c07b"
        elif not self._current_project_id:
            if not self._upload_service._dependencies_ready: # type: ignore
                rag_status_message = "Global RAG: Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                global_size = self._upload_service._vector_db_service.get_collection_size( # type: ignore
                    constants.GLOBAL_COLLECTION_ID)
                if global_size == -1:
                    rag_status_message = "Global RAG: DB Error"
                    rag_status_color = "#e06c75"
                elif global_size == 0:
                    rag_status_message = "Global RAG: Ready (Empty)"
                    rag_status_color = "#e5c07b"
                    current_context_rag_ready = True
                else:
                    rag_status_message = f"Global RAG: Ready ({global_size} chunks)"
                    rag_status_color = "#98c379"
                    current_context_rag_ready = True
        else:
            project_name_display = self._project_manager.get_project_by_id(self._current_project_id)
            project_display_name = project_name_display.name[:15] if project_name_display else self._current_project_id[
                                                                                               :8]

            if not self._upload_service._dependencies_ready: # type: ignore
                rag_status_message = f"RAG for '{project_display_name}...': Dependencies Missing"
                rag_status_color = "#e06c75"
            else:
                project_size = self._upload_service._vector_db_service.get_collection_size(self._current_project_id) # type: ignore
                if project_size == -1:
                    rag_status_message = f"RAG for '{project_display_name}...': DB Error"
                    rag_status_color = "#e06c75"
                elif project_size == 0:
                    rag_status_message = f"RAG for '{project_display_name}...': Ready (Empty)"
                    rag_status_color = "#e5c07b"
                    current_context_rag_ready = True
                else:
                    rag_status_message = f"RAG for '{project_display_name}...': Ready ({project_size} chunks)"
                    rag_status_color = "#98c379"
                    current_context_rag_ready = True

        self._is_rag_ready = current_context_rag_ready
        logger.debug(f"CM: RAG Status: Ready={self._is_rag_ready}, Msg='{rag_status_message}'")
        self._event_bus.ragStatusChanged.emit(self._is_rag_ready, rag_status_message, rag_status_color)

    # Getter methods (optimized with cached values where possible)
    def set_model_for_backend(self, backend_id: str, model_name: str):
        if backend_id == self._active_chat_backend_id:
            if self._active_chat_model_name != model_name:
                self._handle_chat_llm_selection_event(backend_id, model_name)
        elif backend_id == self._active_specialized_backend_id:
            if self._active_specialized_model_name != model_name:
                self._handle_specialized_llm_selection_event(backend_id, model_name)

    def set_chat_temperature(self, temperature: float):
        """✨ NEW: Set chat temperature with validation"""
        if 0.0 <= temperature <= 2.0:
            self._active_temperature = temperature
            self._emit_status_update(f"Temperature: {temperature:.2f}", "#61afef", True, 2000)
            logger.info(f"CM: Temperature set to {temperature:.2f}")
        else:
            logger.warning(f"CM: Invalid temperature {temperature}, must be 0.0-2.0")

    def get_current_chat_history(self) -> List[ChatMessage]:
        return self._current_chat_history

    def get_current_project_id(self) -> Optional[str]:
        return self._current_project_id

    def get_current_session_id(self) -> Optional[str]:
        return self._current_session_id

    def get_current_active_chat_backend_id(self) -> str:
        return self._active_chat_backend_id

    def get_current_active_chat_model_name(self) -> str:
        return self._active_chat_model_name

    def get_current_active_specialized_backend_id(self) -> str:
        return self._active_specialized_backend_id

    def get_current_active_specialized_model_name(self) -> str:
        return self._active_specialized_model_name

    def get_model_for_backend(self, backend_id: str) -> Optional[str]:
        if backend_id == self._active_chat_backend_id:
            return self._active_chat_model_name
        if backend_id == self._active_specialized_backend_id:
            return self._active_specialized_model_name
        return self._backend_coordinator.get_current_configured_model(backend_id)

    def get_all_available_backend_ids(self) -> List[str]:
        return self._backend_coordinator.get_all_backend_ids()

    def get_available_models_for_backend(self, backend_id: str) -> List[str]:
        return self._backend_coordinator.get_available_models_for_backend(backend_id)

    def get_current_chat_personality(self) -> Optional[str]:
        return self._active_chat_personality_prompt

    def get_current_specialized_personality(self) -> Optional[str]:
        return constants.CODER_AI_SYSTEM_PROMPT

    def get_current_chat_temperature(self) -> float:
        return self._active_temperature

    def is_api_ready(self) -> bool:
        return self._is_chat_backend_configured

    def is_specialized_api_ready(self) -> bool:
        return self._is_specialized_backend_configured

    def is_rag_ready(self) -> bool:
        return self._is_rag_ready

    def is_overall_busy(self) -> bool:
        return bool(self._current_llm_request_id) or self._backend_coordinator.is_any_backend_busy()

    def get_llm_communication_logger(self) -> Optional[LlmCommunicationLogger]:
        return self._llm_comm_logger

    def get_backend_coordinator(self) -> BackendCoordinator:
        return self._backend_coordinator

    def get_project_manager(self) -> ProjectManager: # type: ignore
        return self._project_manager

    def get_upload_service(self) -> Optional[UploadService]: # type: ignore
        return self._upload_service

    def get_rag_handler(self) -> Optional[RagHandler]: # type: ignore
        return self._rag_handler

    def cleanup_phase1(self):
        logger.info("ChatManager optimized cleanup...")
        if self._current_llm_request_id:
            self._backend_coordinator.cancel_current_task(self._current_llm_request_id)
            self._current_llm_request_id = None
        self._event_bus.hideLoader.emit()

        # Enhanced cleanup with performance tracking reset
        self._cleanup_code_streams()
        self._response_times.clear()
        self._chunk_counts.clear()

    def _handle_file_creation_request(self, user_msg_txt: str, filename: Optional[str]):
        """Optimized file creation with fast processing."""
        if not self._is_chat_backend_configured:
            self._emit_status_update("Chat backend not configured for file creation.", "#e06c75", True, 3000)
            return

        if not filename:
            filename = self._detect_file_creation_intent(user_msg_txt)
            if not filename:
                asyncio.create_task(self._handle_normal_chat_async(user_msg_txt, []))
                return

        self._event_bus.showLoader.emit(f"Crafting {filename}...")
        logger.info(f"CM: Handling file creation request for '{filename}' using '{self._active_chat_backend_id}'")

        # Use optimized prompt creation
        file_creation_prompt = self._create_file_creation_prompt(user_msg_txt, filename)
        history_for_llm = [ChatMessage(role=USER_ROLE, parts=[file_creation_prompt])]

        success, err, req_id = self._backend_coordinator.initiate_llm_chat_request(self._active_chat_backend_id,
                                                                                   history_for_llm,
                                                                                   {"temperature": 0.2})
        if not success or not req_id:
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[f"[Error creating file: {err}]"])
            self._current_chat_history.append(err_msg_obj)
            if self._current_project_id and self._current_session_id:
                self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id,
                                                          err_msg_obj)
            self._emit_status_update(f"Failed to start file creation: {err or 'Unknown'}", "#FF6B6B")
            self._event_bus.hideLoader.emit()
            return

        # Track timing
        import time
        self._response_times[req_id] = time.time()

        self._file_creation_request_ids[req_id] = filename
        self._current_llm_request_id = req_id
        placeholder = ChatMessage(id=req_id, role=MODEL_ROLE, parts=[""], loading_state=MessageLoadingState.LOADING)
        self._current_chat_history.append(placeholder)
        if self._current_project_id and self._current_session_id:
            self._event_bus.newMessageAddedToHistory.emit(self._current_project_id, self._current_session_id, placeholder)

        self._backend_coordinator.start_llm_streaming_task(req_id, self._active_chat_backend_id, history_for_llm, False,
                                                           {"temperature": 0.2},
                                                           {"purpose": "file_creation",
                                                            "project_id": self._current_project_id,
                                                            "session_id": self._current_session_id,
                                                            "filename": filename})
# core/plan_and_code_coordinator.py
import logging
import uuid
import re
import asyncio
import os
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE, MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from utils import constants
except ImportError as e_pacc:
    logging.getLogger(__name__).critical(f"PlanAndCodeCoordinator: Critical import error: {e_pacc}", exc_info=True)
    # Define fallbacks for parsing
    EventBus = BackendCoordinator = LlmCommunicationLogger = type("Fallback", (object,), {})  # type: ignore
    ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE, MessageLoadingState = (type("Fallback", (object,),
                                                                                             {}),) * 6  # type: ignore
    constants = type("Fallback", (object,), {"CODER_AI_SYSTEM_PROMPT": ""})  # type: ignore
    raise

logger = logging.getLogger(__name__)

MAX_CONCURRENT_CODERS = 3  # This isn't actively used with current task dispatch, but kept for reference
MAX_VALIDATION_RETRIES = 1  # Reduce retries for now to simplify debugging validation flow


class PlanAndCodeCoordinator(QObject):
    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 llm_comm_logger: Optional[LlmCommunicationLogger],
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._llm_comm_logger = llm_comm_logger

        self._active_planning_request_id: Optional[str] = None
        self._active_coder_request_ids: Dict[str, str] = {}  # request_id -> filename
        # self._coder_tasks: List[asyncio.Task] = [] # Not strictly needed with current event-driven completion

        self._current_plan_text: Optional[str] = None
        self._parsed_files_list: List[str] = []
        self._coder_instructions_map: Dict[str, str] = {}  # filename -> instructions
        self._generated_code_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}  # filename -> (code, error_msg)

        # Terminal validation tracking
        self._validation_retry_count: Dict[str, int] = {}  # filename -> retry_count
        self._pending_validation_commands: Dict[str, str] = {}  # command_id -> filename
        self._validation_queue: List[str] = []  # filenames to validate
        self._current_validation_file: Optional[str] = None
        self._project_files_dir: Optional[str] = None

        # Context tracking
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._original_user_query: Optional[str] = None
        self._user_task_type: Optional[str] = None  # NEW: Store detected task type
        self._planner_llm_backend_id: Optional[str] = None  # Store for consistency
        self._planner_llm_model_name: Optional[str] = None  # Store for consistency
        self._specialized_llm_backend_id: Optional[str] = None
        self._specialized_llm_model_name: Optional[str] = None

        self._event_bus.llmResponseCompleted.connect(self._handle_llm_responses)
        self._event_bus.llmResponseError.connect(self._handle_llm_errors)
        self._event_bus.terminalCommandCompleted.connect(self._handle_terminal_command_completed)
        self._event_bus.terminalCommandError.connect(self._handle_terminal_command_error)

        logger.info("PlanAndCodeCoordinator initialized with terminal validation support.")

    def _log_comm(self, sender_prefix: str, message: str):
        if self._llm_comm_logger:
            # log_message = f"[PACC:{sender_prefix}] {message}" # This was redundant, logger auto-prepends
            self._llm_comm_logger.log_message(f"PACC:{sender_prefix}", message)
        else:
            logger.info(f"PACC_LOG_FALLBACK: [PACC:{sender_prefix}] {message[:150]}...")

    def _emit_system_message_to_chat(self, message_text: str, is_error: bool = False,
                                     request_id_ref: Optional[str] = None):
        """Emit a system message to the current chat context"""
        if self._current_project_id and self._current_session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            # Use a consistent ID if this system message is tied to an LLM request, otherwise new UUID
            msg_id = request_id_ref if request_id_ref else uuid.uuid4().hex
            sys_msg = ChatMessage(id=msg_id, role=role, parts=[message_text])  # type: ignore
            self._event_bus.newMessageAddedToHistory.emit(
                self._current_project_id, self._current_session_id, sys_msg
            )
            self._log_comm("CHAT_MSG", f"Sent to UI (Role: {role}): {message_text[:80]}...")
        else:
            logger.warning(f"PACC System Message (No P/S context to emit to UI): {message_text}")

    def _detect_file_task_type(self, filename: str, instructions: str) -> str:
        """Detect task type for individual files based on filename and instructions"""
        filename_lower = filename.lower()
        instructions_lower = instructions.lower()

        # Check filename patterns first
        if any(term in filename_lower for term in ['api', 'server', 'endpoint', 'route', 'handler']):
            return 'api'
        elif any(term in filename_lower for term in ['data', 'process', 'parse', 'etl', 'csv', 'json']):
            return 'data_processing'
        elif any(term in filename_lower for term in ['ui', 'window', 'dialog', 'widget', 'gui', 'interface']):
            return 'ui'
        elif any(term in filename_lower for term in ['util', 'helper', 'tool', 'lib', 'common']):
            return 'utility'
        elif filename_lower in ['main.py', 'app.py', 'run.py', '__main__.py']:
            return 'general'  # Main files usually coordinate everything

        # Check instructions content
        api_keywords = ['endpoint', 'route', 'api', 'fastapi', 'flask', 'http', 'rest', 'server']
        data_keywords = ['data', 'csv', 'json', 'parse', 'process', 'pandas', 'numpy', 'database']
        ui_keywords = ['ui', 'interface', 'window', 'dialog', 'widget', 'pyside', 'qt', 'gui']

        if any(keyword in instructions_lower for keyword in api_keywords):
            return 'api'
        elif any(keyword in instructions_lower for keyword in data_keywords):
            return 'data_processing'
        elif any(keyword in instructions_lower for keyword in ui_keywords):
            return 'ui'

        # Fall back to the project-level task type or general
        return self._user_task_type or 'general'

    def start_planning_sequence(self,
                                user_query: str,
                                planner_llm_backend_id: str,
                                planner_llm_model_name: str,
                                planner_llm_temperature: float,
                                specialized_llm_backend_id: str,
                                specialized_llm_model_name: str,
                                project_files_dir: Optional[str] = None,
                                project_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                user_task_type: Optional[str] = None):  # NEW PARAMETER
        if self._active_planning_request_id or self._active_coder_request_ids:
            logger.warning("PACC: Planning or coding sequence already active. Ignoring new request.")
            self._event_bus.uiStatusUpdateGlobal.emit("Coordinator is already working on a task!", "#e5c07b", True,
                                                      3000)
            return

        self._reset_sequence_state()  # Ensure clean state before starting

        self._original_user_query = user_query
        self._user_task_type = user_task_type  # NEW: Store the detected task type
        self._planner_llm_backend_id = planner_llm_backend_id
        self._planner_llm_model_name = planner_llm_model_name
        self._specialized_llm_backend_id = specialized_llm_backend_id
        self._specialized_llm_model_name = specialized_llm_model_name
        self._project_files_dir = project_files_dir or os.getcwd()
        self._current_project_id = project_id
        self._current_session_id = session_id

        self._active_planning_request_id = f"planner_req_{uuid.uuid4().hex[:12]}"

        self._log_comm("SEQ_START",
                       f"Planning for query: '{user_query[:50]}...' (Task: {user_task_type}, PlannerReqID: {self._active_planning_request_id})")
        self._event_bus.uiStatusUpdateGlobal.emit(f"Asking Planner ({planner_llm_model_name}) to create a plan...",
                                                  "#61afef", False, 0)
        self._event_bus.uiInputBarBusyStateChanged.emit(True)

        # --- ADD PLANNER PLACEHOLDER MESSAGE ---
        if self._current_project_id and self._current_session_id:
            planner_placeholder_text = f"[System: Planner AI ({self._planner_llm_model_name}) is generating a plan for '{self._original_user_query[:30]}...' ⏳]"
            planner_placeholder_msg = ChatMessage(
                id=self._active_planning_request_id,  # Use planner's request ID
                role=MODEL_ROLE,  # Display as an AI response being generated
                parts=[planner_placeholder_text],
                loading_state=MessageLoadingState.LOADING,  # type: ignore
                metadata={"purpose": "plan_and_code_planner_placeholder",
                          "pacc_request_id": self._active_planning_request_id}
            )
            self._event_bus.newMessageAddedToHistory.emit(
                self._current_project_id,
                self._current_session_id,
                planner_placeholder_msg
            )
            self._log_comm("UI_MSG_ADD", f"Planner placeholder sent for ReqID: {self._active_planning_request_id}")
        else:
            logger.warning("PACC: No P/S context, cannot send planner placeholder message to UI.")
        # --- END PLANNER PLACEHOLDER ---

        planner_prompt_text = self._construct_planner_prompt(user_query)
        history_for_planner = [ChatMessage(role=USER_ROLE, parts=[planner_prompt_text])]  # type: ignore

        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._active_planning_request_id,
            target_backend_id=planner_llm_backend_id,
            history_to_send=history_for_planner,
            is_modification_response_expected=True,
            options={"temperature": planner_llm_temperature},
            request_metadata={
                "purpose": "plan_and_code_planner",
                "pacc_request_id": self._active_planning_request_id,
                # Ensure PACC ID is passed for LLM response routing
                "project_id": self._current_project_id,
                "session_id": self._current_session_id
            }
        )

    def _construct_planner_prompt(self, user_query: str) -> str:
        """Construct a task-specific planning prompt"""
        base_prompt = [
            "You are an expert AI system planner and technical architect. Your task is to prepare a comprehensive plan and highly detailed instructions for a separate Coder AI to implement a user's request.",
            f"The user's request is: \"{user_query}\"",
            f"\nProject directory: {self._project_files_dir}",
        ]

        # Add task-specific guidance based on detected task type
        if self._user_task_type == 'api':
            base_prompt.append("\nTASK TYPE: API Development")
            base_prompt.append(
                "Focus on creating RESTful APIs with proper routing, request/response handling, validation, and error handling. Consider using FastAPI or Flask frameworks.")
        elif self._user_task_type == 'data_processing':
            base_prompt.append("\nTASK TYPE: Data Processing")
            base_prompt.append(
                "Focus on efficient data handling, parsing, transformation, and analysis. Consider using pandas, numpy, or appropriate data processing libraries.")
        elif self._user_task_type == 'ui':
            base_prompt.append("\nTASK TYPE: User Interface")
            base_prompt.append(
                "Focus on creating intuitive user interfaces with proper layout, event handling, and user experience. Consider using PySide6/PyQt6 for desktop applications.")
        elif self._user_task_type == 'utility':
            base_prompt.append("\nTASK TYPE: Utility/Helper Functions")
            base_prompt.append(
                "Focus on creating reusable, well-documented utility functions with proper error handling and testing capabilities.")
        else:
            base_prompt.append("\nTASK TYPE: General Development")
            base_prompt.append("Focus on clean, maintainable code with proper structure and best practices.")

        base_prompt.extend([
            "\nYour response MUST be structured as follows:",
            "1.  **Overall Design Philosophy:** Briefly (1-2 sentences) describe the approach for the project structure and main components.",
            "",
            "2.  **Files Planned:** EXACTLY in this format:",
            "FILES_PLANNED: ['relative/path/file1.py', 'relative/path/file2.py', 'relative/path/file3.py']",
            "",
            "IMPORTANT: The FILES_PLANNED line must be EXACTLY as shown above - start with 'FILES_PLANNED:' followed by a Python list.",
            "Example: FILES_PLANNED: ['calculator.py', 'utils.py', 'main.py']",
            "",
            "3.  **Per-File Coder Instructions:** For EACH file in FILES_PLANNED, provide detailed instructions:",
            "",
            "--- CODER_INSTRUCTIONS_START: relative/path/filename.py ---",
            "File Purpose: [Brief description of this file's role.]",
            "Is New File: Yes",  # Or 'No' if modifying existing
            "Inter-File Dependencies: [List other planned files this file interacts with, and how.]",
            "Key Requirements:",
            "- [Detailed instruction 1 for Coder AI. Be explicit: function signatures with type hints, class structures, logic flow, error handling, etc.]",
            "- [Detailed instruction 2...]",
            "- [More detailed instructions...]",
            "Imports Needed: [Suggest specific imports required for this file.]",
            "IMPORTANT CODER OUTPUT FORMAT: The Coder AI MUST respond with ONE single Markdown fenced code block: ```python\\nCODE_HERE\\n```. NO other text.",
            "--- CODER_INSTRUCTIONS_END: relative/path/filename.py ---",
            "",
            "Ensure the Coder AI instructions are thorough enough for a separate, specialized code generation AI to produce complete and correct code for each file.",
            "Focus on clarity, modularity, and best practices. The Coder AI will use a dedicated system prompt focused on code quality (PEP 8, type hints, docstrings, robustness).",
            "",
            "CRITICAL REMINDERS:",
            "- FILES_PLANNED must be a valid Python list format",
            "- Each file needs matching CODER_INSTRUCTIONS_START/END blocks",
            "- Generated code will be automatically validated using linting tools",
            "",
            "Example structure for a calculator request:",
            "FILES_PLANNED: ['calculator.py', 'main.py']"
        ])
        return "\n".join(base_prompt)

    def _parse_planner_response(self, plan_text: str) -> bool:
        self._parsed_files_list = []
        self._coder_instructions_map = {}

        try:
            files_planned_patterns = [
                r"FILES_PLANNED:\s*(\[.*?\])",
                r"Files?\s*Planned:\s*(\[.*?\])",
                r"FILES_PLANNED\s*=\s*(\[.*?\])",
                r"files?_planned:\s*(\[.*?\])",
                r"(?:FILES_PLANNED|Files?\s*Planned).*?(\[.*?\])",
            ]
            files_planned_match = None
            files_list_str = ""  # Initialize to avoid UnboundLocalError

            for pattern in files_planned_patterns:
                files_planned_match = re.search(pattern, plan_text, re.DOTALL | re.IGNORECASE)
                if files_planned_match:
                    logger.info(f"PACC: Found FILES_PLANNED using pattern: {pattern}")
                    files_list_str = files_planned_match.group(
                        1) if files_planned_match.groups() and files_planned_match.group(
                        1) else files_planned_match.group(0)
                    break

            if not files_planned_match:
                logger.error("PACC: Could not find 'FILES_PLANNED:' section in planner response.")
                logger.error(f"PACC: Planner response preview: {plan_text[:500]}...")
                self._log_comm("Parser", "Error: 'FILES_PLANNED:' section not found.")
                # Attempt fallback extraction of any Python list
                fallback_match = re.search(r"\[\s*(?:['\"][^'\"]*['\"],?\s*)*\s*\]", plan_text)
                if fallback_match:
                    logger.warning("PACC: Attempting fallback list extraction for FILES_PLANNED...")
                    files_list_str = fallback_match.group(0)
                else:
                    logger.error("PACC: Fallback list extraction also failed for FILES_PLANNED.")
                    return False

            logger.info(f"PACC: Extracted files list string: {files_list_str}")
            try:
                parsed_list_candidate = eval(files_list_str)  # Use eval carefully, ensure string is somewhat validated
                if not isinstance(parsed_list_candidate, list):
                    logger.error(f"PACC: 'FILES_PLANNED' content is not a list: {files_list_str}")
                    self._log_comm("Parser", f"Error: 'FILES_PLANNED' content not a list: {files_list_str[:100]}...")
                    return False
                self._parsed_files_list = [str(f).strip().replace("\\", "/") for f in parsed_list_candidate if
                                           isinstance(f, str) and f.strip()]
            except Exception as e_eval:
                logger.error(f"PACC: Error evaluating FILES_PLANNED list string '{files_list_str}': {e_eval}",
                             exc_info=True)
                self._log_comm("Parser", f"Error evaluating FILES_PLANNED: {e_eval}")
                return False

            if not self._parsed_files_list:
                logger.info("PACC: Planner indicated no files to be generated in the FILES_PLANNED list.")
                self._log_comm("Parser", "Info: FILES_PLANNED list is empty.")
                return True  # This is a valid outcome

            self._generated_code_map = {fname: (None, None) for fname in self._parsed_files_list}
            self._validation_retry_count = {fname: 0 for fname in self._parsed_files_list}
            logger.info(f"PACC: Parsed planned files: {self._parsed_files_list}")
            self._log_comm("Parser", f"Successfully parsed file list: {self._parsed_files_list}")

            missing_instructions_for_files = []
            for filename in self._parsed_files_list:
                normalized_filename_for_marker = filename.replace("\\", "/")  # Normalize for regex
                instruction_patterns = [
                    # Strict match first
                    f"--- CODER_INSTRUCTIONS_START: {re.escape(normalized_filename_for_marker)} ---(.*?)--- CODER_INSTRUCTIONS_END: {re.escape(normalized_filename_for_marker)} ---",
                    # More lenient if strict fails
                    f"(?:CODER_INSTRUCTIONS_START|Instructions for)\\s*:\\s*{re.escape(normalized_filename_for_marker)}.*?\n(.*?)(?=(?:--- CODER_INSTRUCTIONS_END:|Instructions for|CODER_INSTRUCTIONS_START:|$))"
                ]
                instruction_text = None
                for idx, pattern in enumerate(instruction_patterns):
                    instruction_match = re.search(pattern, plan_text, re.DOTALL | re.IGNORECASE)
                    if instruction_match:
                        instruction_text = instruction_match.group(1).strip()
                        logger.info(f"PACC: Found instructions for {filename} using pattern #{idx + 1}")
                        break

                if instruction_text:
                    self._coder_instructions_map[filename] = instruction_text
                else:
                    logger.warning(f"PACC: Could not find coder instructions for file: {filename} using any pattern.")
                    fallback_instructions = (f"File Purpose: Implementation for {filename}\n"
                                             f"Is New File: Yes\n"
                                             f"Key Requirements:\n"
                                             f"- Implement main functionality for {filename}\n"
                                             f"- Follow Python best practices, type hints, docstrings\n"
                                             f"- Robust error handling\n"
                                             f"- Modular code\n"
                                             f"Imports Needed: Standard Python libraries as needed\n"
                                             f"IMPORTANT CODER OUTPUT FORMAT: Respond with ONE single Markdown fenced code block: ```python\\nCODE_HERE\\n```. NO other text.")
                    self._coder_instructions_map[filename] = fallback_instructions.strip()
                    missing_instructions_for_files.append(filename)

            if missing_instructions_for_files:
                logger.warning(f"PACC: Using fallback instructions for files: {missing_instructions_for_files}")
                self._log_comm("Parser",
                               f"Warning: Using fallback instructions for: {', '.join(missing_instructions_for_files)}")

            return True

        except Exception as e:
            logger.error(f"PACC: Critical error parsing planner response: {e}", exc_info=True)
            self._log_comm("Parser", f"Critical parsing error: {e}")
            return False

    async def _dispatch_code_generation_tasks_async(self):
        if not self._parsed_files_list or not self._coder_instructions_map:
            logger.warning("PACC: No parsed files or instructions to dispatch for code generation.")
            self._emit_system_message_to_chat(
                "[System: Planner did not specify files or instructions. Cannot generate code.]", is_error=True)
            self._event_bus.uiStatusUpdateGlobal.emit("No files or instructions from plan to generate.", "#e5c07b",
                                                      True, 4000)
            self._reset_sequence_state()  # Resets busy state too
            return

        self._log_comm("DISPATCH", f"Starting code generation for {len(self._parsed_files_list)} files.")
        self._emit_system_message_to_chat(
            f"[System: Starting code generation for {len(self._parsed_files_list)} files...]")
        self._event_bus.uiStatusUpdateGlobal.emit(f"Sending {len(self._parsed_files_list)} file(s) to Code LLM...",
                                                  "#61afef", False, 0)
        # Input bar busy state should already be true from planner

        for filename in self._parsed_files_list:
            instructions = self._coder_instructions_map.get(filename)
            if not instructions or instructions.startswith("[Error:"):  # Check for explicit error messages
                logger.warning(f"PACC: Skipping code generation for '{filename}' due to missing/error in instructions.")
                self._generated_code_map[filename] = (None, instructions or "Instructions were missing or invalid.")
                self._emit_system_message_to_chat(
                    f"[System Error: Skipping code generation for `{filename}` due to missing/invalid instructions from planner.]",
                    is_error=True)
                continue

            # Start the code generation task (don't await it here, let event handlers manage completion)
            await self._generate_single_file_code_async(filename, instructions)

        if not self._active_coder_request_ids:  # If no valid tasks were started
            logger.warning("PACC: No valid coder tasks were started after dispatch attempt.")
            self._handle_all_coder_tasks_done()  # This will then lead to final completion

    async def _generate_single_file_code_async(self, filename: str, instructions: str):
        if not self._specialized_llm_backend_id or not self._specialized_llm_model_name:
            logger.error("PACC: Specialized LLM details not set. Cannot generate code for {filename}.")
            self._generated_code_map[filename] = (None, "Specialized LLM (Coder) not configured.")
            self._emit_system_message_to_chat(f"[System Error: Code LLM not configured. Cannot generate `{filename}`.]",
                                              is_error=True)
            # If this is the last file and it fails here, need to ensure sequence finishes.
            # Check if this was the only file or if others are pending.
            if not self._active_coder_request_ids and filename in self._parsed_files_list and self._parsed_files_list.index(
                    filename) == len(self._parsed_files_list) - 1:
                self._handle_all_coder_tasks_done()
            return

        coder_request_id = f"coder_req_{filename.replace('/', '_').replace('.', '_')}_{uuid.uuid4().hex[:8]}"
        self._active_coder_request_ids[coder_request_id] = filename  # Add before starting task

        self._log_comm("CODER_REQ", f"Requesting code for: {filename} (CoderReqID: {coder_request_id})")

        # --- ADD CODER PLACEHOLDER MESSAGE ---
        if self._current_project_id and self._current_session_id:
            coder_placeholder_text = f"[System: Code LLM ({self._specialized_llm_model_name}) is generating code for `{filename}`... ⏳]"
            coder_placeholder_msg = ChatMessage(
                id=coder_request_id,  # Use coder's request ID
                role=MODEL_ROLE,
                parts=[coder_placeholder_text],
                loading_state=MessageLoadingState.LOADING,  # type: ignore
                metadata={"purpose": "plan_and_code_coder_placeholder", "filename": filename,
                          "pacc_request_id": coder_request_id}
            )
            self._event_bus.newMessageAddedToHistory.emit(
                self._current_project_id,
                self._current_session_id,
                coder_placeholder_msg
            )
            self._log_comm("UI_MSG_ADD", f"Coder placeholder sent for {filename}, ReqID: {coder_request_id}")
        else:
            logger.warning(f"PACC: No P/S context, cannot send coder placeholder for {filename} to UI.")
        # --- END CODER PLACEHOLDER ---

        # NEW: Create task-specific prompts for individual files
        file_task_type = self._detect_file_task_type(filename, instructions)

        base_coder_prompt = f"Based on the following instructions, generate the complete Python code for the file `{filename}`.\n\n--- INSTRUCTIONS ---\n{instructions}\n--- END INSTRUCTIONS ---"

        # Add task-specific guidance for the coder
        if file_task_type == 'api':
            task_specific_guidance = f"\n\nTASK-SPECIFIC GUIDANCE:\n{constants.API_DEVELOPMENT_PROMPT}"
        elif file_task_type == 'data_processing':
            task_specific_guidance = f"\n\nTASK-SPECIFIC GUIDANCE:\n{constants.DATA_PROCESSING_PROMPT}"
        elif file_task_type == 'ui':
            task_specific_guidance = f"\n\nTASK-SPECIFIC GUIDANCE:\n{constants.UI_DEVELOPMENT_PROMPT}"
        elif file_task_type == 'utility':
            task_specific_guidance = f"\n\nTASK-SPECIFIC GUIDANCE:\n{constants.UTILITY_DEVELOPMENT_PROMPT}"
        else:
            task_specific_guidance = f"\n\nTASK-SPECIFIC GUIDANCE:\n{constants.GENERAL_CODING_PROMPT}"

        coder_prompt_text = base_coder_prompt + task_specific_guidance
        logger.info(f"PACC: Using {file_task_type} prompt for {filename}")

        history_for_coder = [ChatMessage(role=USER_ROLE, parts=[coder_prompt_text])]  # type: ignore

        self._backend_coordinator.start_llm_streaming_task(
            request_id=coder_request_id,
            target_backend_id=self._specialized_llm_backend_id,
            history_to_send=history_for_coder,
            is_modification_response_expected=True,
            options={"temperature": 0.2},  # Lower temp for more deterministic code
            request_metadata={
                "purpose": "plan_and_code_coder",
                "pacc_request_id": coder_request_id,  # Ensure PACC ID is passed
                "filename": filename,
                "project_id": self._current_project_id,
                "session_id": self._current_session_id
            }
        )
        logger.info(f"PACC: Started code generation task for {filename} (ReqID: {coder_request_id})")

    def _handle_all_coder_tasks_done(self):
        logger.info("PACC: All coder tasks have completed (or errored). Preparing for validation or finalization.")
        self._log_comm("CODERS_DONE", "All individual file generation tasks finished.")

        if self._active_coder_request_ids:  # Should be empty if truly all done
            logger.warning(
                f"PACC: _handle_all_coder_tasks_done called, but {len(self._active_coder_request_ids)} tasks still marked active. This might be a race condition or error. Active IDs: {list(self._active_coder_request_ids.keys())}")
            # Don't proceed if there's a mismatch in state.
            # This might require a timeout mechanism or more robust tracking if it happens often.
            return

        successfully_generated_files_for_validation = []
        for filename in self._parsed_files_list:
            code, err_msg = self._generated_code_map.get(filename, (None, "Result not stored."))
            if code and not err_msg:  # Successfully generated code
                self._write_file_to_disk(filename, code)  # Write before validation
                successfully_generated_files_for_validation.append(filename)
            elif err_msg:
                self._emit_system_message_to_chat(
                    f"[System Error: Code generation failed for `{filename}`: {err_msg[:100]}...]", is_error=True,
                    request_id_ref=filename)  # Use filename as a loose ref ID for UI
            else:  # No code and no error message - implies task might not have run or had an unlogged issue
                self._emit_system_message_to_chat(
                    f"[System Error: Code generation for `{filename}` did not produce output or an error message.]",
                    is_error=True, request_id_ref=filename)

        if successfully_generated_files_for_validation:
            self._log_comm("VALIDATE_START",
                           f"Starting validation for {len(successfully_generated_files_for_validation)} files...")
            self._emit_system_message_to_chat(
                f"[System: Code generation complete. Starting validation for {len(successfully_generated_files_for_validation)} files...]")
            self._event_bus.uiStatusUpdateGlobal.emit(
                f"Code generated for {len(successfully_generated_files_for_validation)} files. Starting validation...",
                "#61afef", False, 0)
            self._validation_queue = successfully_generated_files_for_validation[:]  # Copy
            self._start_next_validation()
        else:
            logger.info("PACC: No files were successfully generated for validation. Moving to final completion.")
            self._emit_system_message_to_chat(
                "[System: No files were successfully generated by Code LLM. Nothing to validate.]", is_error=True)
            self._handle_final_completion()

    def _write_file_to_disk(self, filename: str, content: str):
        if not self._project_files_dir:
            logger.error(f"PACC: Project files directory not set. Cannot write file {filename}.")
            self._log_comm("FILE_WRITE_ERR", f"Project dir not set for {filename}")
            return
        try:
            # Normalize path components to avoid issues with mixed slashes if any
            normalized_filename = os.path.normpath(filename)
            file_path = os.path.join(self._project_files_dir, normalized_filename)

            # Ensure parent directories exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"PACC: Written file to disk: {file_path}")
            self._log_comm("FILE_WRITE_OK", f"Created/Updated: {filename}")
        except Exception as e:
            logger.error(f"PACC: Error writing file {filename} to {file_path}: {e}", exc_info=True)
            self._log_comm("FILE_WRITE_ERR", f"Error writing {filename}: {e}")

    def _start_next_validation(self):
        if not self._validation_queue:
            logger.info("PACC: Validation queue empty. Moving to final completion.")
            self._handle_final_completion()
            return

        self._current_validation_file = self._validation_queue.pop(0)
        self._log_comm("VALIDATE_FILE", f"Validating: {self._current_validation_file}")
        self._emit_system_message_to_chat(f"[System: Validating `{self._current_validation_file}`...]")

        if self._current_validation_file.endswith('.py'):
            self._validate_python_file(self._current_validation_file)
        else:
            logger.info(f"PACC: Skipping validation for non-Python file: {self._current_validation_file}")
            self._log_comm("VALIDATE_SKIP", f"Non-Python file: {self._current_validation_file}")
            self._start_next_validation()  # Move to next

    def _validate_python_file(self, filename: str):
        if not self._project_files_dir:
            logger.error(f"PACC: Project files directory not set. Cannot validate {filename}.")
            self._handle_validation_failure(filename, "Project directory not configured for validation.")
            return

        file_path = os.path.join(self._project_files_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"PACC: File {file_path} not found on disk for validation.")
            self._handle_validation_failure(filename, "File not found on disk for validation.")
            return

        command = f"python -m py_compile \"{file_path}\""  # Ensure path is quoted for spaces
        command_id = f"validate_{uuid.uuid4().hex[:8]}"
        self._pending_validation_commands[command_id] = filename
        self._log_comm("TERM_CMD_REQ", f"Validation cmd for {filename}: {command} (ID: {command_id})")
        self._event_bus.terminalCommandRequested.emit(command, self._project_files_dir, command_id)

    @Slot(str, int, float)
    def _handle_terminal_command_completed(self, command_id: str, exit_code: int, execution_time: float):
        if command_id not in self._pending_validation_commands:
            logger.debug(f"PACC: Terminal command {command_id} completed, but not a pending validation command.")
            return

        filename = self._pending_validation_commands.pop(command_id)
        logger.info(
            f"PACC: Validation command for '{filename}' (ID: {command_id}) completed with exit code {exit_code}.")

        if exit_code == 0:
            self._log_comm("VALIDATE_PASS", f"✓ Validation passed for: {filename}")
            self._emit_system_message_to_chat(f"[System: Validation PASSED for `{filename}`. ✅]")
            self._start_next_validation()
        else:
            self._log_comm("VALIDATE_FAIL", f"✗ Validation failed (exit {exit_code}) for: {filename}")
            self._emit_system_message_to_chat(
                f"[System Error: Validation FAILED for `{filename}` (exit code {exit_code}). See LLM log for details. ❌]",
                is_error=True)
            # Log the terminal output from LLM Communication Logger (it should have received it)
            # For now, just mark as failure and continue
            self._handle_validation_failure(filename, f"py_compile failed with exit code {exit_code}")

    @Slot(str, str)
    def _handle_terminal_command_error(self, command_id: str, error_message: str):
        if command_id not in self._pending_validation_commands:
            logger.debug(f"PACC: Terminal command error for {command_id}, but not a pending validation command.")
            return

        filename = self._pending_validation_commands.pop(command_id)
        logger.error(f"PACC: Validation command for '{filename}' (ID: {command_id}) failed with error: {error_message}")
        self._log_comm("VALIDATE_ERR", f"✗ Validation command error for {filename}: {error_message}")
        self._emit_system_message_to_chat(
            f"[System Error: Validation command for `{filename}` failed: {error_message[:100]}... ❌]", is_error=True)
        self._handle_validation_failure(filename, f"Terminal execution error: {error_message}")

    def _handle_validation_failure(self, filename: str, error_message: str):
        # Currently, we don't retry with LLM fixes. Just log and move on.
        # This could be expanded in the future.
        logger.warning(f"PACC: Validation failed permanently for '{filename}'. Error: {error_message}")
        # Mark this file as having a validation error in _generated_code_map if needed
        if filename in self._generated_code_map:
            code, _ = self._generated_code_map[filename]
            self._generated_code_map[filename] = (code, f"Validation Failed: {error_message}")
        else:  # Should not happen if logic is correct
            self._generated_code_map[filename] = (None, f"Validation Failed (code not found): {error_message}")

        self._start_next_validation()  # Proceed to the next file

    def _handle_final_completion(self):
        logger.info("PACC: All code generation and validation processes complete. Finalizing sequence.")
        self._log_comm("SEQ_FINALIZE", "All generation and validation done.")

        successful_files_final = []
        failed_files_with_errors_final: Dict[str, str] = {}

        for filename in self._parsed_files_list:  # Iterate over initially planned files
            code, err_msg = self._generated_code_map.get(filename, (None, "Generation/validation result not recorded."))

            if code and not err_msg:  # Code exists and no error message (implies validation passed or not applicable)
                successful_files_final.append(filename)
                self._log_comm("FINAL_CODE_OK", f"Finalized code for {filename}.")
                self._event_bus.modificationFileReadyForDisplay.emit(filename, code)
            else:  # Code might exist but there's an error, or code doesn't exist and there's an error
                final_error_msg = err_msg or "Unknown error or missing code."
                failed_files_with_errors_final[filename] = final_error_msg
                self._log_comm("FINAL_CODE_ERR", f"Final error for {filename}: {final_error_msg}")
                # Display even failed/raw code if available, with a note
                if code:  # Code was generated but validation failed or other error
                    self._event_bus.modificationFileReadyForDisplay.emit(filename,
                                                                         f"# VALIDATION FAILED or ERROR: {final_error_msg}\n\n{code}")
                else:  # No code was generated
                    self._event_bus.modificationFileReadyForDisplay.emit(filename,
                                                                         f"# CODE GENERATION FAILED: {final_error_msg}")

        summary_parts = [f"[System: Autonomous coding sequence for '{self._original_user_query[:50]}...' has finished."]
        if successful_files_final:
            summary_parts.append(
                f"Successfully generated/validated: {', '.join(f'`{f}`' for f in successful_files_final)}.")
        if failed_files_with_errors_final:
            failed_details = ", ".join(
                [f"`{f}` ({failed_files_with_errors_final[f][:30]}...)" for f in failed_files_with_errors_final])
            summary_parts.append(f"Issues encountered for: {failed_details}.")
        if not successful_files_final and not failed_files_with_errors_final and self._parsed_files_list:
            summary_parts.append("No files were processed or results are unclear. Check logs.")
        elif not self._parsed_files_list:
            summary_parts.append("Planner did not specify any files to generate.")

        final_status_msg_for_chat = " ".join(summary_parts)
        self._emit_system_message_to_chat(final_status_msg_for_chat, is_error=bool(failed_files_with_errors_final))

        final_ui_status_msg = f"Autonomous coding finished. Success: {len(successful_files_final)}, Issues: {len(failed_files_with_errors_final)}"
        final_ui_status_color = "#56b6c2" if not failed_files_with_errors_final else "#e06c75" if successful_files_final else "#FF6B6B"
        self._event_bus.uiStatusUpdateGlobal.emit(final_ui_status_msg, final_ui_status_color, False, 0)
        self._event_bus.uiInputBarBusyStateChanged.emit(False)
        self._reset_sequence_state()

    def _reset_sequence_state(self):
        logger.info("PACC: Full sequence state reset.")
        self._active_planning_request_id = None
        self._active_coder_request_ids.clear()
        self._current_plan_text = None
        self._parsed_files_list.clear()
        self._coder_instructions_map.clear()
        self._generated_code_map.clear()
        self._validation_retry_count.clear()
        self._pending_validation_commands.clear()
        self._validation_queue.clear()
        self._current_validation_file = None
        self._project_files_dir = None

        # Clear context fields
        self._current_project_id = None
        self._current_session_id = None
        self._original_user_query = None
        self._user_task_type = None  # NEW: Reset task type
        self._planner_llm_backend_id = None
        self._planner_llm_model_name = None
        self._specialized_llm_backend_id = None
        self._specialized_llm_model_name = None

        # Ensure input bar is re-enabled if it was left busy
        # This is now handled by _handle_final_completion emitting uiInputBarBusyStateChanged(False)
        self._log_comm("SEQ_RESET", "Sequence state has been fully reset.")

    @Slot(str, ChatMessage, dict)  # type: ignore
    def _handle_llm_responses(self, request_id: str, completed_message: ChatMessage, usage_stats_dict: dict):
        purpose = usage_stats_dict.get("purpose")
        # pacc_internal_req_id = usage_stats_dict.get("pacc_request_id") # Redundant if request_id is unique

        logger.info(f"PACC: LLM Response RECVD. ReqID: {request_id}, Purpose: {purpose}")
        self._log_comm("LLM_RESP",
                       f"Purpose: {purpose}, ReqID: {request_id}, For Planner: {request_id == self._active_planning_request_id}, For Coder: {request_id in self._active_coder_request_ids}")

        if purpose == "plan_and_code_planner" and request_id == self._active_planning_request_id:
            logger.info(f"PACC: Received PLAN from Planner LLM (ReqID: {request_id})")
            self._current_plan_text = completed_message.text  # type: ignore
            self._log_comm("PLANNER_RECV", f"Plan text length {len(self._current_plan_text or '')} chars.")
            self._active_planning_request_id = None  # Planner task is done

            # Finalize the placeholder message for planner
            self._emit_system_message_to_chat(
                f"[System: Planner AI has provided a plan for '{self._original_user_query[:30]}...'. Parsing now...]",
                request_id_ref=request_id)

            if self._parse_planner_response(self._current_plan_text or ""):
                if not self._parsed_files_list:
                    msg_text = f"[System: Planner LLM indicates no files are needed for '{self._original_user_query[:50]}...']"
                    self._emit_system_message_to_chat(msg_text)
                    self._event_bus.uiStatusUpdateGlobal.emit("Planner indicates no files needed.", "#56b6c2", True,
                                                              3000)
                    self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release busy state
                    self._reset_sequence_state()
                else:
                    files_str = ", ".join([f"`{f}`" for f in self._parsed_files_list])
                    instructions_found_count = sum(1 for f in self._parsed_files_list if
                                                   self._coder_instructions_map.get(
                                                       f) and not self._coder_instructions_map.get(f, "").startswith(
                                                       "[Error:"))
                    all_instructions_ok = instructions_found_count == len(self._parsed_files_list)

                    msg_text = f"[System: Plan parsed for '{self._original_user_query[:50]}...'. Files: {files_str}. Instructions OK: {instructions_found_count}/{len(self._parsed_files_list)}."
                    if all_instructions_ok:
                        msg_text += " Dispatching to Code LLM...]"
                        self._emit_system_message_to_chat(msg_text)
                        asyncio.create_task(self._dispatch_code_generation_tasks_async())
                    else:
                        msg_text += " Some instructions missing/invalid. Aborting code generation.]"
                        self._emit_system_message_to_chat(msg_text, is_error=True)
                        self._event_bus.uiStatusUpdateGlobal.emit("Plan parsed with errors. Code generation aborted.",
                                                                  "#e06c75", False, 0)
                        self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release busy state
                        self._reset_sequence_state()
            else:  # Parsing failed
                err_msg_text = f"[System Error: Failed to parse the plan from Planner LLM for '{self._original_user_query[:50]}...'. Please check LLM logs or try rephrasing.]"
                self._emit_system_message_to_chat(err_msg_text, is_error=True)
                self._event_bus.uiStatusUpdateGlobal.emit("Failed to parse plan.", "#e06c75", False, 0)
                self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release busy state
                self._reset_sequence_state()

        elif purpose == "plan_and_code_coder" and request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.get(request_id, "unknown_file")  # Should always exist
            logger.info(f"PACC: Received CODE from Code LLM for file '{filename}' (ReqID: {request_id})")
            self._log_comm("CODER_RECV", f"Code received for {filename}, ReqID: {request_id}")

            raw_code_response = completed_message.text.strip() if completed_message.text else ""  # type: ignore
            logger.debug(
                f"PACC: Raw code response for {filename} (length: {len(raw_code_response)} chars): '{raw_code_response[:100]}...'")

            # Try to extract only the code block
            # Pattern looks for ``` optional_lang then content then ```
            code_block_match = re.search(r"```(?:[a-zA-Z0-9_\-.+#\s]*\n)?(.*?)```", raw_code_response,
                                         re.DOTALL | re.IGNORECASE)
            extracted_code: Optional[str] = None
            if code_block_match:
                extracted_code = code_block_match.group(1).strip()
                self._generated_code_map[filename] = (extracted_code, None)
                logger.info(f"PACC: Successfully extracted code for {filename} ({len(extracted_code)} chars)")
                self._log_comm("CODE_EXTRACT_OK", f"Extracted code for {filename}")
            else:
                logger.warning(
                    f"PACC: Could not extract fenced code block for '{filename}'. Storing raw response as code.")
                self._log_comm("CODE_EXTRACT_WARN", f"No fenced block for {filename}. Using raw.")
                # If no code block, store the raw response as is, it might be just code.
                # But also flag it as a potential issue for later review if validation fails.
                # Storing raw response IF it looks like code, otherwise mark as error
                if len(raw_code_response) > 0 and (
                        "def " in raw_code_response or "class " in raw_code_response or "import " in raw_code_response or raw_code_response.startswith(
                        "#")):
                    self._generated_code_map[filename] = (raw_code_response,
                                                          "Warning: No fenced code block found, using raw response.")
                else:
                    self._generated_code_map[filename] = (None,
                                                          "Error: No fenced code block found and raw response doesn't look like code.")

            # Finalize the placeholder message for this coder task
            final_coder_msg_text = f"[System: Code LLM finished generating `{filename}`. {'Proceeding...' if extracted_code else 'Error extracting code.'}]"
            self._emit_system_message_to_chat(final_coder_msg_text, is_error=(not extracted_code),
                                              request_id_ref=request_id)

            if request_id in self._active_coder_request_ids:
                del self._active_coder_request_ids[request_id]
            else:  # Should not happen
                logger.warning(
                    f"PACC: Coder ReqID {request_id} for file {filename} was not in _active_coder_request_ids when trying to remove.")

            logger.info(
                f"PACC: Coder task for {filename} completed. Remaining active coder tasks: {len(self._active_coder_request_ids)}")
            if not self._active_coder_request_ids:  # Check if all coder tasks are now done
                logger.info("PACC: All coder tasks seem to be completed. Proceeding to _handle_all_coder_tasks_done.")
                self._handle_all_coder_tasks_done()
        else:
            logger.debug(f"PACC: Ignoring LLM response for unrelated ReqID/Purpose: {request_id} / {purpose}")

    @Slot(str, str)  # type: ignore
    def _handle_llm_errors(self, request_id: str, error_message: str):
        logger.error(f"PACC: LLM Error RECVD. ReqID: {request_id}, Error: {error_message}")
        self._log_comm("LLM_ERROR", f"ReqID: {request_id}, Error: {error_message[:100]}...")

        if request_id == self._active_planning_request_id:
            logger.error(f"PACC: Error from Planner LLM (ReqID: {request_id}): {error_message}")
            self._active_planning_request_id = None  # Planner task failed
            error_msg_text = f"[System Error: Planner LLM failed to generate a plan for '{self._original_user_query[:50]}...': {error_message}]"
            self._emit_system_message_to_chat(error_msg_text, is_error=True, request_id_ref=request_id)
            self._event_bus.uiStatusUpdateGlobal.emit("Planner LLM error. Unable to create plan.", "#e06c75", False, 0)
            self._event_bus.uiInputBarBusyStateChanged.emit(False)  # Release busy state
            self._reset_sequence_state()

        elif request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.pop(request_id, "unknown_file_on_error")
            logger.error(f"PACC: Error from Code LLM for file '{filename}' (ReqID: {request_id}): {error_message}")
            self._generated_code_map[filename] = (None, error_message)  # Store error
            error_msg_text = f"[System Error: Code LLM failed for file `{filename}`: {error_message}]"
            self._emit_system_message_to_chat(error_msg_text, is_error=True, request_id_ref=request_id)

            logger.info(
                f"PACC: Coder task for {filename} errored. Remaining active coder tasks: {len(self._active_coder_request_ids)}")
            if not self._active_coder_request_ids:  # Check if all coder tasks are now done (including this error)
                logger.info(
                    "PACC: All coder tasks seem to be completed (with errors). Proceeding to _handle_all_coder_tasks_done.")
                self._handle_all_coder_tasks_done()
        else:
            logger.debug(f"PACC: Ignoring LLM error for unrelated ReqID: {request_id}")
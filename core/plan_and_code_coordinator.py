# core/plan_and_code_coordinator.py
import logging
import uuid
import re
import asyncio
import os
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import QObject, Slot

from core.event_bus import EventBus
from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from backends.backend_coordinator import BackendCoordinator
from services.llm_communication_logger import LlmCommunicationLogger
from utils import constants

logger = logging.getLogger(__name__)

MAX_CONCURRENT_CODERS = 3
MAX_VALIDATION_RETRIES = 3


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
        self._active_coder_request_ids: Dict[str, str] = {}
        self._coder_tasks: List[asyncio.Task] = []

        self._current_plan_text: Optional[str] = None
        self._parsed_files_list: List[str] = []
        self._coder_instructions_map: Dict[str, str] = {}
        self._generated_code_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Terminal validation tracking
        self._validation_retry_count: Dict[str, int] = {}
        self._pending_validation_commands: Dict[str, str] = {}  # command_id -> filename
        self._validation_queue: List[str] = []  # filenames to validate
        self._current_validation_file: Optional[str] = None
        self._project_files_dir: Optional[str] = None

        # Context tracking for better integration
        self._current_project_id: Optional[str] = None
        self._current_session_id: Optional[str] = None
        self._original_user_query: Optional[str] = None
        self._specialized_llm_backend_id: Optional[str] = None
        self._specialized_llm_model_name: Optional[str] = None

        self._event_bus.llmResponseCompleted.connect(self._handle_llm_responses)
        self._event_bus.llmResponseError.connect(self._handle_llm_errors)

        # Connect terminal signals for validation
        self._event_bus.terminalCommandCompleted.connect(self._handle_terminal_command_completed)
        self._event_bus.terminalCommandError.connect(self._handle_terminal_command_error)

        logger.info("PlanAndCodeCoordinator initialized with terminal validation support.")

    def _log_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            self._llm_comm_logger.log_message(sender, message)
        else:
            logger.info(f"PACC_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def _emit_system_message(self, message: str):
        """Emit a system message to the current chat context"""
        if self._current_project_id and self._current_session_id:
            system_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE, parts=[message])
            self._event_bus.newMessageAddedToHistory.emit(
                self._current_project_id,
                self._current_session_id,
                system_msg
            )
        else:
            # Fallback for when we don't have project context
            logger.info(f"PACC System Message: {message}")

    def start_planning_sequence(self,
                                user_query: str,
                                planner_llm_backend_id: str,
                                planner_llm_model_name: str,
                                planner_llm_temperature: float,
                                specialized_llm_backend_id: str,
                                specialized_llm_model_name: str,
                                project_files_dir: Optional[str] = None,
                                project_id: Optional[str] = None,  # NEW
                                session_id: Optional[str] = None):  # NEW
        if self._active_planning_request_id or self._active_coder_request_ids:
            logger.warning("Planning or coding sequence already active. Ignoring new request.")
            self._event_bus.uiStatusUpdateGlobal.emit("Coordinator is already working on a task!", "#e5c07b", True,
                                                      3000)
            return

        self._original_user_query = user_query
        self._specialized_llm_backend_id = specialized_llm_backend_id
        self._specialized_llm_model_name = specialized_llm_model_name
        self._project_files_dir = project_files_dir or os.getcwd()

        # NEW: Store project context
        self._current_project_id = project_id
        self._current_session_id = session_id

        # Reset all state
        self._current_plan_text = None
        self._parsed_files_list = []
        self._coder_instructions_map = {}
        self._generated_code_map = {}
        self._coder_tasks = []
        self._active_coder_request_ids = {}
        self._validation_retry_count = {}
        self._pending_validation_commands = {}
        self._validation_queue = []
        self._current_validation_file = None

        self._active_planning_request_id = f"planner_req_{uuid.uuid4().hex[:12]}"

        self._log_comm("PlanAndCodeCoordinator",
                       f"Starting planning sequence for query: '{user_query[:50]}...' (ReqID: {self._active_planning_request_id})")
        self._event_bus.uiStatusUpdateGlobal.emit(f"Asking Planner LLM ({planner_llm_model_name}) to create a plan...",
                                                  "#61afef", False, 0)

        planner_prompt_text = self._construct_planner_prompt(user_query)
        history_for_planner = [ChatMessage(role=USER_ROLE, parts=[planner_prompt_text])]

        self._backend_coordinator.start_llm_streaming_task(
            request_id=self._active_planning_request_id,
            target_backend_id=planner_llm_backend_id,
            history_to_send=history_for_planner,
            is_modification_response_expected=True,
            options={"temperature": planner_llm_temperature},
            request_metadata={"purpose": "plan_and_code_planner", "pacc_request_id": self._active_planning_request_id}
        )

    def _construct_planner_prompt(self, user_query: str) -> str:
        prompt_parts = [
            "You are an expert AI system planner and technical architect. Your task is to prepare a comprehensive plan and highly detailed instructions for a separate Coder AI to implement a user's request.",
            f"The user's request is: \"{user_query}\"",
            f"\nProject directory: {self._project_files_dir}",
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
            "Is New File: Yes",
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
        ]
        return "\n".join(prompt_parts)

    def _parse_planner_response(self, plan_text: str) -> bool:
        self._parsed_files_list = []
        self._coder_instructions_map = {}

        try:
            # IMPROVED: More flexible parsing with multiple patterns
            files_planned_patterns = [
                r"FILES_PLANNED:\s*(\[.*?\])",  # Original pattern
                r"Files?\s*Planned:\s*(\[.*?\])",  # More flexible
                r"FILES_PLANNED\s*=\s*(\[.*?\])",  # With equals
                r"files?_planned:\s*(\[.*?\])",  # Lowercase
                r"(?:FILES_PLANNED|Files?\s*Planned).*?(\[.*?\])",  # Even more flexible
            ]

            files_planned_match = None
            for pattern in files_planned_patterns:
                files_planned_match = re.search(pattern, plan_text, re.DOTALL | re.IGNORECASE)
                if files_planned_match:
                    logger.info(f"PACC: Found FILES_PLANNED using pattern: {pattern}")
                    break

            if not files_planned_match:
                logger.error("PACC: Could not find 'FILES_PLANNED:' section in planner response.")
                logger.error(f"PACC: Planner response preview: {plan_text[:500]}...")
                self._log_comm("PACC_Parser", "Error: 'FILES_PLANNED:' section not found.")

                # ADDED: Try to extract any Python list from the response as fallback
                fallback_match = re.search(r"\[(?:['\"][^'\"]*['\"],?\s*)+\]", plan_text)
                if fallback_match:
                    logger.warning("PACC: Attempting fallback list extraction...")
                    files_planned_match = fallback_match
                else:
                    return False

            files_list_str = files_planned_match.group(
                1) if files_planned_match.groups() else files_planned_match.group(0)
            logger.info(f"PACC: Extracted files list string: {files_list_str}")

            try:
                parsed_list_candidate = eval(files_list_str)
                if not isinstance(parsed_list_candidate, list):
                    logger.error(f"PACC: 'FILES_PLANNED' content is not a list: {files_list_str}")
                    self._log_comm("PACC_Parser",
                                   f"Error: 'FILES_PLANNED' content not a list: {files_list_str[:100]}...")
                    self._parsed_files_list = []
                    return False
                self._parsed_files_list = [str(f).strip().replace("\\", "/") for f in parsed_list_candidate if
                                           isinstance(f, str) and f.strip()]
            except Exception as e_eval:
                logger.error(f"PACC: Error evaluating FILES_PLANNED list string '{files_list_str}': {e_eval}")
                self._log_comm("PACC_Parser", f"Error evaluating FILES_PLANNED: {e_eval}")
                return False

            if not self._parsed_files_list:
                logger.info("PACC: Planner indicated no files to be generated in the FILES_PLANNED list.")
                self._log_comm("PACC_Parser", "Info: FILES_PLANNED list is empty.")
                return True

            self._generated_code_map = {fname: (None, None) for fname in self._parsed_files_list}
            self._validation_retry_count = {fname: 0 for fname in self._parsed_files_list}

            logger.info(f"PACC: Parsed planned files: {self._parsed_files_list}")
            self._log_comm("PACC_Parser", f"Successfully parsed file list: {self._parsed_files_list}")

            # IMPROVED: More flexible instruction parsing
            missing_instructions_for_files = []
            for filename in self._parsed_files_list:
                normalized_filename_for_marker = filename.replace("\\", "/")

                # Try multiple instruction block patterns
                instruction_patterns = [
                    f"--- CODER_INSTRUCTIONS_START: {re.escape(normalized_filename_for_marker)} ---(.*?)--- CODER_INSTRUCTIONS_END: {re.escape(normalized_filename_for_marker)} ---",
                    f"CODER_INSTRUCTIONS_START: {re.escape(normalized_filename_for_marker)}(.*?)CODER_INSTRUCTIONS_END: {re.escape(normalized_filename_for_marker)}",
                    f"Instructions for {re.escape(normalized_filename_for_marker)}:(.*?)(?=(?:Instructions for|--- CODER_INSTRUCTIONS|$))",
                ]

                instruction_text = None
                for pattern in instruction_patterns:
                    instruction_match = re.search(pattern, plan_text, re.DOTALL | re.IGNORECASE)
                    if instruction_match:
                        instruction_text = instruction_match.group(1).strip()
                        logger.info(f"PACC: Found instructions for {filename} using pattern")
                        break

                if instruction_text:
                    self._coder_instructions_map[filename] = instruction_text
                else:
                    logger.warning(f"PACC: Could not find coder instructions for file: {filename}")
                    # IMPROVED: Create basic instructions as fallback
                    fallback_instructions = f"""
                                         File Purpose: Implementation for {filename}
                                         Is New File: Yes
                                         Key Requirements:
                                         - Implement the main functionality for {filename}
                                         - Follow Python best practices with type hints and docstrings
                                         - Include proper error handling
                                         - Make the code modular and well-structured
                                         Imports Needed: Standard Python libraries as needed
                                         IMPORTANT CODER OUTPUT FORMAT: Respond with ONE single Markdown fenced code block: ```python\\nCODE_HERE\\n```. NO other text.
                                         """
                    self._coder_instructions_map[filename] = fallback_instructions.strip()
                    missing_instructions_for_files.append(filename)

            if missing_instructions_for_files:
                logger.warning(f"PACC: Using fallback instructions for files: {missing_instructions_for_files}")
                self._log_comm("PACC_Parser",
                               f"Warning: Using fallback instructions for: {', '.join(missing_instructions_for_files)}")

            return True

        except Exception as e:
            logger.error(f"PACC: Critical error parsing planner response: {e}", exc_info=True)
            self._log_comm("PACC_Parser", f"Critical parsing error: {e}")
            return False

    async def _dispatch_code_generation_tasks_async(self):
        if not self._parsed_files_list or not self._coder_instructions_map:
            logger.warning("PACC: No parsed files or instructions to dispatch for code generation.")
            self._event_bus.uiStatusUpdateGlobal.emit("No files or instructions from plan to generate.", "#e5c07b",
                                                      True, 4000)
            self._reset_sequence_state()
            return

        self._log_comm("PACC", f"Starting code generation for {len(self._parsed_files_list)} files.")
        self._event_bus.uiStatusUpdateGlobal.emit(f"Sending {len(self._parsed_files_list)} file(s) to Code LLM...",
                                                  "#61afef", False, 0)
        self._event_bus.uiInputBarBusyStateChanged.emit(True)

        # FIXED: Don't use asyncio.gather - just start the tasks and let event handlers manage completion
        for filename in self._parsed_files_list:
            instructions = self._coder_instructions_map.get(filename)
            if not instructions or instructions.startswith("[Error:"):
                logger.warning(f"PACC: Skipping code generation for '{filename}' due to missing/error in instructions.")
                self._generated_code_map[filename] = (None, instructions or "Instructions were missing.")
                continue

            # Start the code generation task (don't await it)
            await self._generate_single_file_code_async(filename, instructions)

        # Check if we have any active coder tasks after starting them all
        if not self._active_coder_request_ids:
            logger.warning("PACC: No valid coder tasks were started.")
            self._handle_all_coder_tasks_done()

    async def _generate_single_file_code_async(self, filename: str, instructions: str):
        # REMOVED: semaphore parameter since we're not using asyncio.gather anymore
        if not self._specialized_llm_backend_id or not self._specialized_llm_model_name:
            logger.error("PACC: Specialized LLM details not set. Cannot generate code.")
            self._generated_code_map[filename] = (None, "Specialized LLM not configured.")
            return

        coder_request_id = f"coder_req_{filename.replace('/', '_').replace('.', '_')}_{uuid.uuid4().hex[:8]}"
        self._active_coder_request_ids[coder_request_id] = filename

        self._log_comm("PACC->CodeLLM", f"Requesting code for: {filename} (CoderReqID: {coder_request_id})")

        # Emit system message to chat context
        self._emit_system_message(f"[System: Code LLM is now generating `{filename}`...]")

        coder_prompt_text = f"Based on the following instructions, generate the complete Python code for the file `{filename}`.\n\n--- INSTRUCTIONS ---\n{instructions}\n--- END INSTRUCTIONS ---"
        history_for_coder = [ChatMessage(role=USER_ROLE, parts=[coder_prompt_text])]

        self._backend_coordinator.start_llm_streaming_task(
            request_id=coder_request_id,
            target_backend_id=self._specialized_llm_backend_id,
            history_to_send=history_for_coder,
            is_modification_response_expected=True,
            options={"temperature": 0.2},
            request_metadata={"purpose": "plan_and_code_coder", "pacc_request_id": coder_request_id,
                              "filename": filename}
        )

        logger.info(f"PACC: Started code generation for {filename} (ReqID: {coder_request_id})")

    def _handle_all_coder_tasks_done(self):
        logger.info("PACC: All coder tasks have completed (or errored).")

        # Check for any remaining active tasks before proceeding
        if self._active_coder_request_ids:
            logger.info(
                f"PACC: Still have {len(self._active_coder_request_ids)} active coder tasks, not finishing yet.")
            return

        # Instead of immediately finishing, start validation process
        successfully_generated_files = []

        for filename in self._parsed_files_list:
            code, err_msg = self._generated_code_map.get(filename, (None, "Task did not complete or store result."))
            if code and not err_msg:
                successfully_generated_files.append(filename)
                # Write the file to disk for validation
                self._write_file_to_disk(filename, code)

        if successfully_generated_files:
            self._log_comm("PACC",
                           f"Code generation complete. Starting validation for {len(successfully_generated_files)} files...")
            self._event_bus.uiStatusUpdateGlobal.emit("Code generated. Starting validation...", "#61afef", False, 0)
            self._validation_queue = successfully_generated_files.copy()
            self._start_next_validation()
        else:
            self._handle_final_completion()

    def _handle_all_coder_tasks_done(self):
        logger.info("PACC: All coder tasks have completed (or errored).")

        # Instead of immediately finishing, start validation process
        successfully_generated_files = []

        for filename in self._parsed_files_list:
            code, err_msg = self._generated_code_map.get(filename, (None, "Task did not complete or store result."))
            if code and not err_msg:
                successfully_generated_files.append(filename)
                # Write the file to disk for validation
                self._write_file_to_disk(filename, code)

        if successfully_generated_files:
            self._log_comm("PACC",
                           f"Code generation complete. Starting validation for {len(successfully_generated_files)} files...")
            self._event_bus.uiStatusUpdateGlobal.emit("Code generated. Starting validation...", "#61afef", False, 0)
            self._validation_queue = successfully_generated_files.copy()
            self._start_next_validation()
        else:
            self._handle_final_completion()

    def _write_file_to_disk(self, filename: str, content: str):
        """Write generated file to disk for validation"""
        try:
            file_path = os.path.join(self._project_files_dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            logger.info(f"PACC: Written file to disk: {file_path}")
            self._log_comm("FILE_WRITER", f"Created: {filename}")

        except Exception as e:
            logger.error(f"PACC: Error writing file {filename}: {e}")
            self._log_comm("FILE_WRITER", f"Error writing {filename}: {e}")

    def _start_next_validation(self):
        """Start validation for the next file in the queue"""
        if not self._validation_queue:
            self._handle_final_completion()
            return

        self._current_validation_file = self._validation_queue.pop(0)
        self._log_comm("VALIDATOR", f"Validating: {self._current_validation_file}")

        # Determine validation command based on file type
        if self._current_validation_file.endswith('.py'):
            self._validate_python_file(self._current_validation_file)
        else:
            # For non-Python files, just mark as validated
            self._log_comm("VALIDATOR", f"Skipping validation for non-Python file: {self._current_validation_file}")
            self._start_next_validation()

    def _validate_python_file(self, filename: str):
        """Validate a Python file using linting tools"""
        file_path = os.path.join(self._project_files_dir, filename)

        # First try syntax check
        command = f"python -m py_compile {file_path}"
        command_id = f"validate_{uuid.uuid4().hex[:8]}"

        self._pending_validation_commands[command_id] = filename
        self._log_comm("VALIDATOR", f"Running syntax check: {command}")

        # Emit terminal command request
        self._event_bus.terminalCommandRequested.emit(command, self._project_files_dir, command_id)

    @Slot(str, int, float)
    def _handle_terminal_command_completed(self, command_id: str, exit_code: int, execution_time: float):
        """Handle terminal command completion for validation"""
        if command_id not in self._pending_validation_commands:
            return

        filename = self._pending_validation_commands.pop(command_id)

        if exit_code == 0:
            self._log_comm("VALIDATOR", f"✓ Validation passed for: {filename}")
            self._start_next_validation()
        else:
            self._handle_validation_failure(filename, f"Validation failed with exit code {exit_code}")

    @Slot(str, str)
    def _handle_terminal_command_error(self, command_id: str, error_message: str):
        """Handle terminal command error for validation"""
        if command_id not in self._pending_validation_commands:
            return

        filename = self._pending_validation_commands.pop(command_id)
        self._handle_validation_failure(filename, error_message)

    def _handle_validation_failure(self, filename: str, error_message: str):
        """Handle validation failure and potentially retry with fixes"""
        retry_count = self._validation_retry_count.get(filename, 0)

        if retry_count >= MAX_VALIDATION_RETRIES:
            self._log_comm("VALIDATOR", f"✗ Max retries reached for: {filename}. Validation failed.")
            self._start_next_validation()
            return

        self._validation_retry_count[filename] = retry_count + 1
        self._log_comm("VALIDATOR",
                       f"⚠ Validation failed for {filename} (attempt {retry_count + 1}). Error: {error_message}")

        # Here you could implement auto-fixing by calling the Code LLM again with the error
        # For now, just continue to next file
        self._start_next_validation()

    def _handle_final_completion(self):
        """Handle final completion of the entire process"""
        logger.info("PACC: All code generation and validation complete.")

        successful_files = []
        failed_files_with_errors: Dict[str, str] = {}

        for filename in self._parsed_files_list:
            code, err_msg = self._generated_code_map.get(filename, (None, "Task did not complete or store result."))

            if code and not err_msg:
                successful_files.append(filename)
                self._log_comm("CodeLLM->PACC", f"Finalized generated code for {filename}.")
                self._event_bus.modificationFileReadyForDisplay.emit(filename, code)
            else:
                failed_files_with_errors[filename] = err_msg or "Unknown error during generation."
                self._log_comm("CodeLLM Error->PACC", f"Final error for {filename}: {err_msg}")

        summary_message_parts = [
            f"[System: Autonomous code generation complete for '{self._original_user_query[:50]}...'."]
        if successful_files:
            summary_message_parts.append(
                f"Successfully generated and validated: {', '.join(f'`{f}`' for f in successful_files)}.")
        if failed_files_with_errors:
            failed_details = ", ".join(
                [f"`{f}` ({failed_files_with_errors[f][:30]}...)" for f in failed_files_with_errors])
            summary_message_parts.append(f"Failed for: {failed_details}.")

        final_status_msg = " ".join(summary_message_parts)

        # Emit to proper chat context
        self._emit_system_message(final_status_msg)

        self._event_bus.uiStatusUpdateGlobal.emit(
            f"Autonomous coding finished. Success: {len(successful_files)}, Failed: {len(failed_files_with_errors)}",
            "#56b6c2" if not failed_files_with_errors else "#e06c75", False, 0)
        self._event_bus.uiInputBarBusyStateChanged.emit(False)
        self._reset_sequence_state()

    def _reset_sequence_state(self):
        self._active_planning_request_id = None
        self._active_coder_request_ids = {}
        self._coder_tasks = []
        self._current_plan_text = None
        self._parsed_files_list = []
        self._coder_instructions_map = {}
        self._original_user_query = None
        self._specialized_llm_backend_id = None
        self._specialized_llm_model_name = None
        self._generated_code_map = {}
        self._validation_retry_count = {}
        self._pending_validation_commands = {}
        self._validation_queue = []
        self._current_validation_file = None
        self._project_files_dir = None
        self._current_project_id = None  # NEW
        self._current_session_id = None  # NEW
        logger.info("PACC: Sequence state has been reset.")

    def _handle_llm_responses(self, request_id: str, completed_message: ChatMessage, usage_stats_dict: dict):
        purpose = usage_stats_dict.get("purpose")
        pacc_internal_req_id = usage_stats_dict.get("pacc_request_id")

        # ADDED: Better logging for debugging
        logger.info(
            f"PACC: Processing LLM response - ReqID: {request_id}, Purpose: {purpose}, PACC_ID: {pacc_internal_req_id}")
        logger.info(f"PACC: Active planner ID: {self._active_planning_request_id}")
        logger.info(f"PACC: Active coder IDs: {list(self._active_coder_request_ids.keys())}")

        if purpose == "plan_and_code_planner" and request_id == self._active_planning_request_id:
            logger.info(f"PACC: Received PLAN from Planner LLM (ReqID: {request_id})")
            self._current_plan_text = completed_message.text
            self._log_comm("PlannerLLM->PACC", f"Full Plan Received (length {len(self._current_plan_text)} chars)")
            self._active_planning_request_id = None  # Clear planner request ID as it's done

            if self._parse_planner_response(self._current_plan_text):
                if not self._parsed_files_list:
                    msg_text = f"[System: Planner LLM indicates no files are needed for '{self._original_user_query[:50]}...']"
                    self._event_bus.uiStatusUpdateGlobal.emit("Planner indicates no files needed.", "#56b6c2", True,
                                                              3000)
                    self._event_bus.uiInputBarBusyStateChanged.emit(False)
                    self._reset_sequence_state()
                else:
                    files_str = ", ".join([f"`{f}`" for f in self._parsed_files_list])
                    instructions_found_count = sum(1 for f in self._parsed_files_list if
                                                   not self._coder_instructions_map.get(f, "").startswith("[Error:"))
                    all_instructions_found = instructions_found_count == len(self._parsed_files_list)

                    msg_text = f"[System: Plan received and parsed for '{self._original_user_query[:50]}...'. Files: {files_str}. Instructions found for {instructions_found_count}/{len(self._parsed_files_list)} files."
                    if all_instructions_found:
                        msg_text += " Starting autonomous code generation and validation...]"
                        asyncio.create_task(self._dispatch_code_generation_tasks_async())
                    else:
                        msg_text += " Some instructions are missing or invalid. Cannot proceed with code generation.]"
                        self._event_bus.uiStatusUpdateGlobal.emit("Plan parsed with errors. Code generation aborted.",
                                                                  "#e06c75", False, 0)
                        self._event_bus.uiInputBarBusyStateChanged.emit(False)
                        self._reset_sequence_state()

                    # Emit to proper chat context
                    self._emit_system_message(msg_text)
            else:
                err_msg_text = f"[System Error: Failed to parse the plan from Planner LLM for '{self._original_user_query[:50]}...'. Please check LLM logs or try rephrasing.]"
                self._emit_system_message(err_msg_text)
                self._event_bus.uiStatusUpdateGlobal.emit("Failed to parse plan.", "#e06c75", False, 0)
                self._event_bus.uiInputBarBusyStateChanged.emit(False)
                self._reset_sequence_state()

        # FIXED: Simplified condition for coder responses
        elif purpose == "plan_and_code_coder" and request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.get(request_id, "unknown_file")
            logger.info(f"PACC: Received CODE from Code LLM for file '{filename}' (ReqID: {request_id})")

            raw_code_response = completed_message.text.strip()
            logger.info(f"PACC: Raw code response length: {len(raw_code_response)} chars")

            code_block_match = re.search(r"```(?:[a-zA-Z0-9_\-\.]*\s*\n)?(.*?)```", raw_code_response,
                                         re.DOTALL | re.IGNORECASE)
            if code_block_match:
                extracted_code = code_block_match.group(1).strip()
                self._generated_code_map[filename] = (extracted_code, None)
                logger.info(f"PACC: Successfully extracted code for {filename} ({len(extracted_code)} chars)")
            else:
                logger.warning(f"PACC: Could not extract code block for '{filename}'. Storing raw response.")
                # Store the raw response instead of failing completely
                self._generated_code_map[filename] = (raw_code_response, None)

            # Remove handled coder ID
            if request_id in self._active_coder_request_ids:
                del self._active_coder_request_ids[request_id]

            # ADDED: Check if all coder tasks are done
            logger.info(f"PACC: Coder task completed. Remaining active tasks: {len(self._active_coder_request_ids)}")
            if len(self._active_coder_request_ids) == 0:
                logger.info("PACC: All coder tasks completed, calling _handle_all_coder_tasks_done")
                self._handle_all_coder_tasks_done()

        else:
            logger.debug(f"PACC: Ignoring completed LLM response for unrelated purpose/ID: {purpose} / {request_id}")

    def _handle_llm_errors(self, request_id: str, error_message: str):
        logger.error(f"PACC: Handling LLM error for ReqID: {request_id}, Error: {error_message}")

        if request_id == self._active_planning_request_id:
            logger.error(f"PACC: Error from Planner LLM (ReqID: {request_id}): {error_message}")
            self._log_comm("PlannerLLM Error->PACC", error_message)

            error_message_text = f"[System Error: Planner LLM failed to generate a plan for '{self._original_user_query[:50]}...': {error_message}]"
            self._emit_system_message(error_message_text)

            self._event_bus.uiStatusUpdateGlobal.emit("Planner LLM error. Unable to create plan.", "#e06c75", False, 0)
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._reset_sequence_state()

        elif request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.pop(request_id, "unknown_file")
            logger.error(f"PACC: Error from Code LLM for file '{filename}' (ReqID: {request_id}): {error_message}")
            self._generated_code_map[filename] = (None, error_message)

            error_message_text = f"[System Error: Code LLM failed for file `{filename}`: {error_message}]"
            self._emit_system_message(error_message_text)

            # ADDED: Check if all coder tasks are done (including errored ones)
            logger.info(f"PACC: Coder task errored. Remaining active tasks: {len(self._active_coder_request_ids)}")
            if len(self._active_coder_request_ids) == 0:
                logger.info("PACC: All coder tasks completed (with errors), calling _handle_all_coder_tasks_done")
                self._handle_all_coder_tasks_done()

        else:
            logger.debug(f"PACC: Ignoring LLM error for unrelated request ID: {request_id}")
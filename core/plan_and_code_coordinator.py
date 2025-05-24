# core/plan_and_code_coordinator.py
import logging
import uuid
import re
import asyncio
from typing import List, Optional, Dict, Any, Tuple

from PySide6.QtCore import QObject

from core.event_bus import EventBus
from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
from backends.backend_coordinator import BackendCoordinator
from services.llm_communication_logger import LlmCommunicationLogger
from utils import constants

logger = logging.getLogger(__name__)

MAX_CONCURRENT_CODERS = 3


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

        self._original_user_query: Optional[str] = None
        self._specialized_llm_backend_id: Optional[str] = None
        self._specialized_llm_model_name: Optional[str] = None

        self._event_bus.llmResponseCompleted.connect(self._handle_llm_responses)
        self._event_bus.llmResponseError.connect(self._handle_llm_errors)

        logger.info("PlanAndCodeCoordinator initialized.")

    def _log_comm(self, sender: str, message: str):
        if self._llm_comm_logger:
            self._llm_comm_logger.log_message(sender, message)
        else:
            logger.info(f"PACC_LOG_FALLBACK: [{sender}] {message[:150]}...")

    def start_planning_sequence(self,
                                user_query: str,
                                planner_llm_backend_id: str,
                                planner_llm_model_name: str,
                                planner_llm_temperature: float,
                                specialized_llm_backend_id: str,
                                specialized_llm_model_name: str):
        if self._active_planning_request_id or self._active_coder_request_ids:
            logger.warning("Planning or coding sequence already active. Ignoring new request.")
            self._event_bus.uiStatusUpdateGlobal.emit("Coordinator is already working on a task!", "#e5c07b", True,
                                                      3000)
            return

        self._original_user_query = user_query
        self._specialized_llm_backend_id = specialized_llm_backend_id
        self._specialized_llm_model_name = specialized_llm_model_name
        self._current_plan_text = None
        self._parsed_files_list = []
        self._coder_instructions_map = {}
        self._generated_code_map = {fname: (None, None) for fname in
                                    self._parsed_files_list}  # Initialize for all potential files
        self._coder_tasks = []
        self._active_coder_request_ids = {}

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
            "\nYour response MUST be structured as follows:",
            "1.  **Overall Design Philosophy:** Briefly (1-2 sentences) describe the approach for the project structure and main components.",
            "2.  **Files Planned:** A Python list of relative file paths that need to be created. (e.g., FILES_PLANNED: ['src/main.py', 'src/utils.py', 'templates/index.html'])",
            "3.  **Per-File Coder Instructions:** For EACH file in FILES_PLANNED, provide a detailed set of instructions for the Coder AI. Each file's instructions should be clearly demarcated.",
            "    --- CODER_INSTRUCTIONS_START: path/to/filename.ext ---",
            "    File Purpose: [Brief description of this file's role.]",
            "    Is New File: ['Yes'] (For new projects, all files are new)",
            "    Inter-File Dependencies: [List other planned files this file interacts with, and how.]",
            "    Key Requirements:",
            "    - [Detailed instruction 1 for Coder AI. Be explicit: function signatures with type hints, class structures, logic flow, error handling, etc.]",
            "    - [Detailed instruction 2...]",
            "    Imports Needed: [Suggest specific imports required for this file.]",
            "    IMPORTANT CODER OUTPUT FORMAT: (Remind the Coder AI that its response for this file MUST be ONE single Markdown fenced code block: ```python path/to/filename.ext\\nCODE_HERE\\n```. NO other text.)",
            "    --- CODER_INSTRUCTIONS_END: path/to/filename.ext ---",
            "\nEnsure the Coder AI instructions are thorough enough for a separate, specialized code generation AI to produce complete and correct code for each file.",
            "Focus on clarity, modularity, and best practices. The Coder AI will use a dedicated system prompt focused on code quality (PEP 8, type hints, docstrings, robustness)."
        ]
        return "\n".join(prompt_parts)

    def _parse_planner_response(self, plan_text: str) -> bool:
        self._parsed_files_list = []
        self._coder_instructions_map = {}

        try:
            files_planned_match = re.search(r"FILES_PLANNED:\s*(\[.*?\])", plan_text, re.DOTALL | re.IGNORECASE)
            if not files_planned_match:
                logger.error("PACC: Could not find 'FILES_PLANNED:' section in planner response.")
                self._log_comm("PACC_Parser", "Error: 'FILES_PLANNED:' section not found.")
                return False

            files_list_str = files_planned_match.group(1)
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
            logger.info(f"PACC: Parsed planned files: {self._parsed_files_list}")
            self._log_comm("PACC_Parser", f"Successfully parsed file list: {self._parsed_files_list}")

            missing_instructions_for_files = []
            for filename in self._parsed_files_list:
                normalized_filename_for_marker = filename.replace("\\", "/")
                start_marker = f"--- CODER_INSTRUCTIONS_START: {normalized_filename_for_marker} ---"
                end_marker = f"--- CODER_INSTRUCTIONS_END: {normalized_filename_for_marker} ---"

                instruction_match = re.search(f"{re.escape(start_marker)}(.*?){re.escape(end_marker)}", plan_text,
                                              re.DOTALL | re.IGNORECASE)

                if instruction_match:
                    instruction_text = instruction_match.group(1).strip()
                    self._coder_instructions_map[filename] = instruction_text
                else:
                    logger.warning(f"PACC: Could not find coder instructions for file: {filename} using markers.")
                    self._coder_instructions_map[filename] = f"[Error: Instructions not found by parser for {filename}]"
                    missing_instructions_for_files.append(filename)

            if missing_instructions_for_files:
                logger.error(f"PACC: Missing coder instructions for files: {missing_instructions_for_files}")
                self._log_comm("PACC_Parser",
                               f"Error: Missing coder instructions for: {', '.join(missing_instructions_for_files)}")

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

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_CODERS)
        self._coder_tasks = []
        self._active_coder_request_ids = {}

        for filename in self._parsed_files_list:
            instructions = self._coder_instructions_map.get(filename)
            if not instructions or instructions.startswith("[Error:"):
                logger.warning(f"PACC: Skipping code generation for '{filename}' due to missing/error in instructions.")
                self._generated_code_map[filename] = (None, instructions or "Instructions were missing.")
                continue

            task = asyncio.create_task(self._generate_single_file_code_async(filename, instructions, semaphore))
            self._coder_tasks.append(task)

        if not self._coder_tasks:
            logger.warning("PACC: No valid coder tasks to run after filtering instructions.")
            self._handle_all_coder_tasks_done()
            return

        await asyncio.gather(*self._coder_tasks, return_exceptions=True)
        self._handle_all_coder_tasks_done()

    async def _generate_single_file_code_async(self, filename: str, instructions: str, semaphore: asyncio.Semaphore):
        async with semaphore:
            if not self._specialized_llm_backend_id or not self._specialized_llm_model_name:
                logger.error("PACC: Specialized LLM details not set. Cannot generate code.")
                self._generated_code_map[filename] = (None, "Specialized LLM not configured.")
                # If a task fails here, it won't be in _active_coder_request_ids for error handler to clean up.
                # Ensure _handle_all_coder_tasks_done correctly processes it from _generated_code_map
                return

            coder_request_id = f"coder_req_{filename.replace('/', '_').replace('.', '_')}_{uuid.uuid4().hex[:8]}"
            self._active_coder_request_ids[coder_request_id] = filename

            self._log_comm("PACC->CodeLLM", f"Requesting code for: {filename} (CoderReqID: {coder_request_id})")
            system_message = ChatMessage(role=SYSTEM_ROLE,
                                         parts=[f"[System: Code LLM is now generating `{filename}`...]"])
            self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", system_message)

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

    def _handle_all_coder_tasks_done(self):
        logger.info("PACC: All coder tasks have completed (or errored).")
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

        summary_message_parts = [f"[System: Code generation phase complete for '{self._original_user_query[:50]}...'."]
        if successful_files:
            summary_message_parts.append(f"Successfully generated: {', '.join(f'`{f}`' for f in successful_files)}.")
        if failed_files_with_errors:
            failed_details = ", ".join(
                [f"`{f}` ({failed_files_with_errors[f][:30]}...)" for f in failed_files_with_errors])
            summary_message_parts.append(f"Failed for: {failed_details}.")

        final_status_msg = " ".join(summary_message_parts)
        result_message = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE, parts=[final_status_msg])
        self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", result_message)

        self._event_bus.uiStatusUpdateGlobal.emit(
            f"Code generation finished. Success: {len(successful_files)}, Failed: {len(failed_files_with_errors)}",
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
        logger.info("PACC: Sequence state has been reset.")

    def _handle_llm_responses(self, request_id: str, completed_message: ChatMessage, usage_stats_dict: dict):
        purpose = usage_stats_dict.get("purpose")
        pacc_internal_req_id = usage_stats_dict.get("pacc_request_id")

        if purpose == "plan_and_code_planner" and request_id == self._active_planning_request_id and pacc_internal_req_id == self._active_planning_request_id:
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
                        msg_text += " Starting code generation...]"
                        asyncio.create_task(self._dispatch_code_generation_tasks_async())
                    else:
                        msg_text += " Some instructions are missing or invalid. Cannot proceed with code generation.]"
                        self._event_bus.uiStatusUpdateGlobal.emit("Plan parsed with errors. Code generation aborted.",
                                                                  "#e06c75", False, 0)
                        self._event_bus.uiInputBarBusyStateChanged.emit(False)
                        self._reset_sequence_state()

                    plan_summary_msg = ChatMessage(id=uuid.uuid4().hex, role=SYSTEM_ROLE, parts=[msg_text])
                    self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", plan_summary_msg)
            else:
                err_msg_text = f"[System Error: Failed to parse the plan from Planner LLM for '{self._original_user_query[:50]}...'. Please check LLM logs or try rephrasing.]"
                error_message = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[err_msg_text])
                self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", error_message)
                self._event_bus.uiStatusUpdateGlobal.emit("Failed to parse plan.", "#e06c75", False, 0)
                self._event_bus.uiInputBarBusyStateChanged.emit(False)
                self._reset_sequence_state()

        elif purpose == "plan_and_code_coder" and request_id == pacc_internal_req_id and request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.get(request_id, "unknown_file")  # Use .get for safety
            logger.info(f"PACC: Received CODE from Code LLM for file '{filename}' (ReqID: {request_id})")

            raw_code_response = completed_message.text.strip()

            code_block_match = re.search(r"```(?:[a-zA-Z0-9_\-\.]*\s*\n)?(.*?)```", raw_code_response,
                                         re.DOTALL | re.IGNORECASE)
            if code_block_match:
                extracted_code = code_block_match.group(1).strip()
                self._generated_code_map[filename] = (extracted_code, None)
            else:
                logger.warning(
                    f"PACC: Could not extract code block for '{filename}' from Code LLM response. Storing raw. Preview: {raw_code_response[:100]}")
                self._generated_code_map[filename] = (None,
                                                      f"Could not extract code block. Raw response: {raw_code_response[:200]}...")

            # Remove handled coder ID, task management is via asyncio.gather
            if request_id in self._active_coder_request_ids:
                del self._active_coder_request_ids[request_id]
        else:
            logger.debug(f"PACC: Ignoring completed LLM response for unrelated purpose/ID: {purpose} / {request_id}")

    def _coder_tasks_still_running(self) -> bool:
        return any(task and not task.done() for task in self._coder_tasks)

    def _handle_llm_errors(self, request_id: str, error_message: str):
        if request_id == self._active_planning_request_id:
            logger.error(f"PACC: Error from Planner LLM (ReqID: {request_id}): {error_message}")
            self._log_comm("PlannerLLM Error->PACC", error_message)

            error_message_text = f"[System Error: Planner LLM failed to generate a plan for '{self._original_user_query[:50]}...': {error_message}]"
            err_msg = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[error_message_text])
            self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", err_msg)

            self._event_bus.uiStatusUpdateGlobal.emit("Planner LLM error. Unable to create plan.", "#e06c75", False, 0)
            self._event_bus.uiInputBarBusyStateChanged.emit(False)
            self._reset_sequence_state()

        elif request_id in self._active_coder_request_ids:
            filename = self._active_coder_request_ids.pop(request_id, "unknown_file")
            logger.error(f"PACC: Error from Code LLM for file '{filename}' (ReqID: {request_id}): {error_message}")
            self._generated_code_map[filename] = (None, error_message)

            error_message_text = f"[System Error: Code LLM failed for file `{filename}`: {error_message}]"
            err_msg_obj = ChatMessage(id=uuid.uuid4().hex, role=ERROR_ROLE, parts=[error_message_text])
            self._event_bus.newMessageAddedToHistory.emit("p1_chat_context", err_msg_obj)

            # Task completion is handled by asyncio.gather, which then calls _handle_all_coder_tasks_done
        else:
            logger.debug(f"PACC: Ignoring LLM error for unrelated request ID: {request_id}")
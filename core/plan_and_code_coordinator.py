# core/plan_and_code_coordinator.py - Streamlined version
import logging
import uuid
import asyncio
import os
from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from core.models import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE, MessageLoadingState
    from backends.backend_coordinator import BackendCoordinator
    from services.llm_communication_logger import LlmCommunicationLogger
    from core.code_output_processor import CodeOutputProcessor, CodeQualityLevel
    from utils import constants
except ImportError as e_pacc:
    logging.getLogger(__name__).critical(f"PlanAndCodeCoordinator: Critical import error: {e_pacc}", exc_info=True)
    raise

logger = logging.getLogger(__name__)


class SequencePhase(Enum):
    IDLE = auto()
    PLANNING = auto()
    CODE_GENERATION = auto()
    VALIDATION = auto()
    FINALIZATION = auto()


@dataclass
class FileGenerationTask:
    filename: str
    instructions: str
    task_type: str
    request_id: Optional[str] = None
    generated_code: Optional[str] = None
    code_quality: Optional[CodeQualityLevel] = None
    processing_notes: List[str] = field(default_factory=list)
    validation_passed: bool = False
    error_message: Optional[str] = None


class PlanAndCodeCoordinator(QObject):
    """
    Streamlined coordinator focusing on orchestration rather than processing.
    Heavy lifting is delegated to specialized processors.
    """

    def __init__(self,
                 backend_coordinator: BackendCoordinator,
                 event_bus: EventBus,
                 llm_comm_logger: Optional[LlmCommunicationLogger],
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._backend_coordinator = backend_coordinator
        self._event_bus = event_bus
        self._llm_comm_logger = llm_comm_logger

        # Processors
        self._code_processor = CodeOutputProcessor()

        # State management
        self._current_phase = SequencePhase.IDLE
        self._sequence_id: Optional[str] = None
        self._planning_request_id: Optional[str] = None
        self._active_generation_tasks: Dict[str, FileGenerationTask] = {}  # request_id -> task

        # Context
        self._original_query: Optional[str] = None
        self._project_context = {}
        self._plan_text: Optional[str] = None
        self._file_tasks: List[FileGenerationTask] = []

        # Connect to events
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_completion)
        self._event_bus.llmResponseError.connect(self._handle_llm_error)
        self._event_bus.terminalCommandCompleted.connect(self._handle_validation_complete)
        self._event_bus.terminalCommandError.connect(self._handle_validation_error)

        logger.info("Streamlined PlanAndCodeCoordinator initialized")

    def start_autonomous_coding(self,
                                user_query: str,
                                planner_backend: str,
                                planner_model: str,
                                coder_backend: str,
                                coder_model: str,
                                project_dir: str,
                                project_id: Optional[str] = None,
                                session_id: Optional[str] = None,
                                task_type: Optional[str] = None) -> bool:
        """Start the autonomous coding sequence."""

        if self._current_phase != SequencePhase.IDLE:
            logger.warning("Sequence already active, ignoring new request")
            self._emit_status("Already processing a request", "#e5c07b", True, 3000)
            return False

        # Initialize sequence
        self._sequence_id = f"seq_{uuid.uuid4().hex[:8]}"
        self._current_phase = SequencePhase.PLANNING
        self._original_query = user_query

        # Store context
        self._project_context = {
            'project_dir': project_dir,
            'project_id': project_id,
            'session_id': session_id,
            'task_type': task_type,
            'planner_backend': planner_backend,
            'planner_model': planner_model,
            'coder_backend': coder_backend,
            'coder_model': coder_model
        }

        logger.info(f"Starting autonomous coding sequence {self._sequence_id} for: {user_query[:50]}...")
        self._log_comm("SEQ_START", f"Query: {user_query[:100]}...")

        # Start planning phase
        return self._start_planning_phase()

    def _start_planning_phase(self) -> bool:
        """Start the planning phase with the planner LLM."""
        try:
            self._planning_request_id = f"plan_{self._sequence_id}"

            # Send status updates
            self._emit_status(f"Planning with {self._project_context['planner_model']}...", "#61afef", False)
            self._emit_chat_message(f"[System: Starting autonomous coding for '{self._original_query[:30]}...']")
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

            # Create planning prompt
            planning_prompt = self._create_planning_prompt()
            history = [ChatMessage(role=USER_ROLE, parts=[planning_prompt])]

            # Send to backend
            self._backend_coordinator.start_llm_streaming_task(
                request_id=self._planning_request_id,
                target_backend_id=self._project_context['planner_backend'],
                history_to_send=history,
                is_modification_response_expected=True,
                options={"temperature": 0.3},
                request_metadata={
                    "purpose": "autonomous_planning",
                    "sequence_id": self._sequence_id,
                    "project_id": self._project_context.get('project_id'),
                    "session_id": self._project_context.get('session_id')
                }
            )

            return True

        except Exception as e:
            logger.error(f"Failed to start planning phase: {e}", exc_info=True)
            self._handle_sequence_error(f"Planning initialization failed: {e}")
            return False

    def _create_planning_prompt(self) -> str:
        """Create a focused planning prompt."""
        task_guidance = self._get_task_specific_guidance()

        return f"""You are an expert software architect. Create a detailed plan for implementing this request:

REQUEST: {self._original_query}

PROJECT DIRECTORY: {self._project_context['project_dir']}
TASK TYPE: {self._project_context.get('task_type', 'general')}

{task_guidance}

Your response MUST follow this EXACT format:

## Architecture Overview
[Brief 2-3 sentence description of the approach]

## Files Required
FILES_LIST: ['file1.py', 'file2.py', 'file3.py']

## Implementation Details

### file1.py
PURPOSE: [What this file does]
DEPENDENCIES: [Other files it imports/uses] 
REQUIREMENTS:
- [Specific requirement 1]
- [Specific requirement 2]
- [More requirements...]

### file2.py
PURPOSE: [What this file does]
DEPENDENCIES: [Other files it imports/uses]
REQUIREMENTS:
- [Specific requirement 1]
- [Specific requirement 2]

[Continue for each file...]

CRITICAL: The FILES_LIST must be a valid Python list format.
Each file must have a matching implementation section."""

    def _get_task_specific_guidance(self) -> str:
        """Get guidance based on detected task type."""
        task_type = self._project_context.get('task_type', 'general')

        guidance_map = {
            'api': "Focus on RESTful APIs with proper routing, validation, and error handling.",
            'data_processing': "Focus on efficient data handling with pandas/numpy and proper error handling.",
            'ui': "Focus on clean UI design with proper event handling and user experience.",
            'utility': "Focus on reusable functions with comprehensive error handling and documentation.",
            'general': "Focus on clean, maintainable code following Python best practices."
        }

        return guidance_map.get(task_type, guidance_map['general'])

    @Slot(str, object, dict)
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):
        """Handle LLM response completion."""
        purpose = metadata.get("purpose")
        sequence_id = metadata.get("sequence_id")

        if sequence_id != self._sequence_id:
            return  # Not our sequence

        if purpose == "autonomous_planning" and request_id == self._planning_request_id:
            self._handle_planning_complete(message.text)
        elif purpose == "autonomous_coding" and request_id in self._active_generation_tasks:
            self._handle_code_generation_complete(request_id, message.text)

    def _handle_planning_complete(self, plan_text: str):
        """Handle completion of planning phase."""
        logger.info("Planning phase completed, parsing plan...")
        self._plan_text = plan_text
        self._planning_request_id = None

        try:
            # Parse the plan into file tasks
            self._file_tasks = self._parse_plan_response(plan_text)

            if not self._file_tasks:
                self._handle_sequence_error("No files specified in plan")
                return

            logger.info(f"Plan parsed successfully: {len(self._file_tasks)} files to generate")
            self._emit_chat_message(
                f"[System: Plan created for {len(self._file_tasks)} files. Starting code generation...]")

            # Move to code generation phase
            self._current_phase = SequencePhase.CODE_GENERATION
            self._start_code_generation_phase()

        except Exception as e:
            logger.error(f"Failed to parse plan: {e}", exc_info=True)
            self._handle_sequence_error(f"Plan parsing failed: {e}")

    def _parse_plan_response(self, plan_text: str) -> List[FileGenerationTask]:
        """Parse the plan response into file generation tasks."""
        import re

        tasks = []

        # Extract files list
        files_match = re.search(r'FILES_LIST:\s*(\[.*?\])', plan_text, re.DOTALL)
        if not files_match:
            raise ValueError("FILES_LIST not found in plan")

        try:
            files_list = eval(files_match.group(1))  # Careful evaluation
            if not isinstance(files_list, list):
                raise ValueError("FILES_LIST is not a valid list")
        except Exception as e:
            raise ValueError(f"Failed to parse FILES_LIST: {e}")

        # Extract implementation details for each file
        for filename in files_list:
            # Find the implementation section
            section_pattern = rf'### {re.escape(filename)}\s*\n(.*?)(?=\n### |\Z)'
            section_match = re.search(section_pattern, plan_text, re.DOTALL)

            if section_match:
                section_content = section_match.group(1).strip()
                instructions = self._extract_file_instructions(section_content, filename)
            else:
                # Fallback instructions
                instructions = f"Implement {filename} according to the requirements in the plan."

            task = FileGenerationTask(
                filename=filename,
                instructions=instructions,
                task_type=self._detect_file_task_type(filename, instructions)
            )
            tasks.append(task)

        return tasks

    def _extract_file_instructions(self, section_content: str, filename: str) -> str:
        """Extract detailed instructions for a file from its plan section."""
        lines = section_content.split('\n')

        purpose = ""
        dependencies = ""
        requirements = []

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('PURPOSE:'):
                purpose = line[8:].strip()
                current_section = 'purpose'
            elif line.startswith('DEPENDENCIES:'):
                dependencies = line[13:].strip()
                current_section = 'dependencies'
            elif line.startswith('REQUIREMENTS:'):
                current_section = 'requirements'
            elif line.startswith('- ') and current_section == 'requirements':
                requirements.append(line[2:].strip())

        # Build comprehensive instructions
        instructions = f"""File: {filename}
Purpose: {purpose}

Dependencies: {dependencies}

Detailed Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

Implementation Notes:
- Follow Python best practices (PEP 8, type hints, docstrings)
- Include comprehensive error handling
- Add logging where appropriate
- Ensure code is production-ready

OUTPUT FORMAT: Respond with ONLY a Python code block:
```python
[YOUR COMPLETE CODE HERE]
```"""

        return instructions

    def _detect_file_task_type(self, filename: str, instructions: str) -> str:
        """Detect task type for a specific file."""
        filename_lower = filename.lower()
        instructions_lower = instructions.lower()

        # File name patterns
        patterns = {
            'api': ['api', 'server', 'endpoint', 'route', 'handler'],
            'data_processing': ['data', 'process', 'parse', 'etl', 'csv', 'json'],
            'ui': ['ui', 'window', 'dialog', 'widget', 'gui', 'interface'],
            'utility': ['util', 'helper', 'tool', 'lib', 'common']
        }

        for task_type, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return task_type
            if any(keyword in instructions_lower for keyword in keywords):
                return task_type

        return 'general'

    def _start_code_generation_phase(self):
        """Start generating code for all files."""
        self._emit_status(f"Generating code for {len(self._file_tasks)} files...", "#61afef", False)

        # Start generation for all files (they'll be processed as they complete)
        for task in self._file_tasks:
            self._start_single_file_generation(task)

    def _start_single_file_generation(self, task: FileGenerationTask):
        """Start code generation for a single file."""
        task.request_id = f"code_{self._sequence_id}_{task.filename.replace('.', '_').replace('/', '_')}"
        self._active_generation_tasks[task.request_id] = task

        # Create coding prompt
        coding_prompt = self._create_coding_prompt(task)
        history = [ChatMessage(role=USER_ROLE, parts=[coding_prompt])]

        # Send to backend
        self._backend_coordinator.start_llm_streaming_task(
            request_id=task.request_id,
            target_backend_id=self._project_context['coder_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.1},  # Low temperature for code generation
            request_metadata={
                "purpose": "autonomous_coding",
                "sequence_id": self._sequence_id,
                "filename": task.filename,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

        logger.info(f"Started code generation for {task.filename} (request: {task.request_id})")

    def _create_coding_prompt(self, task: FileGenerationTask) -> str:
        """Create a focused coding prompt for a specific file."""
        return f"""You are a Python code generation specialist. Generate complete, working code for:

FILE: {task.filename}
TASK TYPE: {task.task_type}

INSTRUCTIONS:
{task.instructions}

CRITICAL REQUIREMENTS:
1. Respond with ONLY a single Python code block
2. Use this EXACT format: ```python\\n[CODE]\\n```
3. NO explanatory text outside the code block
4. Include all necessary imports at the top
5. Add proper docstrings and type hints
6. Include comprehensive error handling
7. Follow PEP 8 style guidelines

The code must be complete and ready to run."""

    def _handle_code_generation_complete(self, request_id: str, raw_response: str):
        """Handle completion of code generation for a file."""
        task = self._active_generation_tasks.get(request_id)
        if not task:
            logger.warning(f"No task found for request {request_id}")
            return

        logger.info(f"Code generation completed for {task.filename}")

        # Process the response using our specialized processor
        extracted_code, quality, notes = self._code_processor.process_llm_response(
            raw_response, task.filename, "python"
        )

        # Update task with results
        task.generated_code = extracted_code
        task.code_quality = quality
        task.processing_notes = notes

        if extracted_code and quality in [CodeQualityLevel.EXCELLENT, CodeQualityLevel.GOOD,
                                          CodeQualityLevel.ACCEPTABLE]:
            # Write to disk
            self._write_file_to_disk(task.filename, extracted_code)
            logger.info(f"Successfully generated {task.filename} (Quality: {quality.name})")
        else:
            task.error_message = f"Code extraction failed: {', '.join(notes)}"
            logger.warning(f"Failed to generate {task.filename}: {task.error_message}")

        # Clean up
        del self._active_generation_tasks[request_id]

        # Check if all tasks are complete
        if not self._active_generation_tasks:
            self._handle_all_generation_complete()

    def _write_file_to_disk(self, filename: str, content: str):
        """Write generated file to disk."""
        try:
            project_dir = self._project_context['project_dir']
            file_path = os.path.join(project_dir, filename)

            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Clean and format the code
            clean_content = self._code_processor.clean_and_format_code(content)

            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_content)

            logger.info(f"Wrote file: {file_path}")

        except Exception as e:
            logger.error(f"Failed to write {filename}: {e}", exc_info=True)

    def _handle_all_generation_complete(self):
        """Handle completion of all code generation tasks."""
        logger.info("All code generation tasks completed")

        # Count results
        successful = [t for t in self._file_tasks if t.generated_code and not t.error_message]
        failed = [t for t in self._file_tasks if t.error_message or not t.generated_code]

        # Send files to code viewer
        for task in successful:
            if task.generated_code:
                self._event_bus.modificationFileReadyForDisplay.emit(task.filename, task.generated_code)

        # Create summary
        summary_parts = [
            f"[System: Autonomous coding completed for '{self._original_query[:50]}...']",
            f"Successfully generated: {len(successful)} files",
        ]

        if successful:
            files_list = ", ".join(f"`{t.filename}`" for t in successful)
            summary_parts.append(f"Files: {files_list}")

        if failed:
            summary_parts.append(f"Failed: {len(failed)} files")

        self._emit_chat_message(" ".join(summary_parts))

        # Emit final status
        color = "#98c379" if not failed else "#e5c07b" if successful else "#FF6B6B"
        self._emit_status(f"Generated {len(successful)}/{len(self._file_tasks)} files", color, False)

        # Reset state
        self._reset_sequence()

    @Slot(str, str)
    def _handle_llm_error(self, request_id: str, error_message: str):
        """Handle LLM errors."""
        if request_id == self._planning_request_id:
            self._handle_sequence_error(f"Planning failed: {error_message}")
        elif request_id in self._active_generation_tasks:
            task = self._active_generation_tasks[request_id]
            task.error_message = f"Generation failed: {error_message}"
            del self._active_generation_tasks[request_id]

            if not self._active_generation_tasks:
                self._handle_all_generation_complete()

    def _handle_sequence_error(self, error_message: str):
        """Handle sequence-level errors."""
        logger.error(f"Sequence error: {error_message}")
        self._emit_chat_message(f"[System Error: {error_message}]", is_error=True)
        self._emit_status(f"Autonomous coding failed: {error_message}", "#FF6B6B", False)
        self._reset_sequence()

    def _reset_sequence(self):
        """Reset sequence state."""
        self._current_phase = SequencePhase.IDLE
        self._sequence_id = None
        self._planning_request_id = None
        self._active_generation_tasks.clear()
        self._original_query = None
        self._project_context.clear()
        self._plan_text = None
        self._file_tasks.clear()

        self._event_bus.uiInputBarBusyStateChanged.emit(False)
        logger.info("Sequence state reset")

    # Validation methods (simplified placeholders)
    @Slot(str, int, float)
    def _handle_validation_complete(self, command_id: str, exit_code: int, execution_time: float):
        """Handle validation completion - placeholder for future implementation."""
        pass

    @Slot(str, str)
    def _handle_validation_error(self, command_id: str, error_message: str):
        """Handle validation errors - placeholder for future implementation."""
        pass

    # Helper methods
    def _log_comm(self, prefix: str, message: str):
        """Log communication."""
        if self._llm_comm_logger:
            self._llm_comm_logger.log_message(f"PACC:{prefix}", message)

    def _emit_status(self, message: str, color: str, temporary: bool = False, duration: int = 0):
        """Emit status update."""
        self._event_bus.uiStatusUpdateGlobal.emit(message, color, temporary, duration)

    def _emit_chat_message(self, message: str, is_error: bool = False):
        """Emit chat message."""
        project_id = self._project_context.get('project_id')
        session_id = self._project_context.get('session_id')

        if project_id and session_id:
            role = ERROR_ROLE if is_error else SYSTEM_ROLE
            chat_msg = ChatMessage(role=role, parts=[message])
            self._event_bus.newMessageAddedToHistory.emit(project_id, session_id, chat_msg)

    def is_busy(self) -> bool:
        """Check if coordinator is busy."""
        return self._current_phase != SequencePhase.IDLE
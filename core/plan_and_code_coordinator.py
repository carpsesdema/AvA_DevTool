# core/plan_and_code_coordinator.py - FIXED for reliable multi-file generation
import logging
import uuid
import asyncio
import os
import re
from typing import List, Optional, Dict, Any, Set, Tuple
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
    DEPENDENCY_ANALYSIS = auto()
    CODE_GENERATION = auto()
    VALIDATION = auto()
    FINALIZATION = auto()


class GenerationStrategy(Enum):
    SEQUENTIAL = auto()  # One file at a time
    BATCHED = auto()  # Small batches based on dependencies
    PARALLEL = auto()  # All at once (risky)


@dataclass
class FileGenerationTask:
    filename: str
    instructions: str
    task_type: str
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    generation_order: int = 0
    request_id: Optional[str] = None
    generated_code: Optional[str] = None
    code_quality: Optional[CodeQualityLevel] = None
    processing_notes: List[str] = field(default_factory=list)
    validation_passed: bool = False
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class GenerationBatch:
    """A batch of files that can be generated in parallel (no dependencies between them)"""
    files: List[FileGenerationTask]
    batch_number: int
    completed_count: int = 0
    failed_count: int = 0


class PlanAndCodeCoordinator(QObject):
    """
    FIXED: Robust multi-file code generation with dependency resolution and error recovery.
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

        # FIXED: Better tracking for multi-file generation
        self._generation_batches: List[GenerationBatch] = []
        self._current_batch_index = 0
        self._current_generation_request_id: Optional[str] = None
        self._generated_files_context: Dict[str, str] = {}  # filename -> content

        # Context
        self._original_query: Optional[str] = None
        self._project_context = {}
        self._plan_text: Optional[str] = None
        self._file_tasks: List[FileGenerationTask] = []

        # Configuration
        self._generation_strategy = GenerationStrategy.SEQUENTIAL  # Start with safest approach
        self._max_batch_size = 3  # Maximum files per batch
        self._generation_delay = 2.0  # Delay between generations to avoid overwhelming LLM

        # Connect to events
        self._event_bus.llmResponseCompleted.connect(self._handle_llm_completion)
        self._event_bus.llmResponseError.connect(self._handle_llm_error)

        logger.info("FIXED PlanAndCodeCoordinator initialized with robust multi-file generation")

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
        """Start the autonomous coding sequence with robust multi-file support."""

        if self._current_phase != SequencePhase.IDLE:
            logger.warning("Sequence already active, ignoring new request")
            self._emit_status("Already processing a request", "#e5c07b", True, 3000)
            return False

        # Initialize sequence
        self._sequence_id = f"seq_{uuid.uuid4().hex[:8]}"
        self._current_phase = SequencePhase.PLANNING
        self._original_query = user_query

        # Reset state for multi-file generation
        self._generation_batches.clear()
        self._current_batch_index = 0
        self._current_generation_request_id = None
        self._generated_files_context.clear()

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

        logger.info(f"Starting ROBUST autonomous coding sequence {self._sequence_id} for: {user_query[:50]}...")
        self._log_comm("SEQ_START_ROBUST", f"Query: {user_query[:100]}...")

        # Start planning phase
        return self._start_planning_phase()

    def _start_planning_phase(self) -> bool:
        """Start the planning phase with enhanced prompts for multi-file projects."""
        try:
            self._planning_request_id = f"plan_{self._sequence_id}"

            # Send status updates
            self._emit_status(f"Planning with {self._project_context['planner_model']}...", "#61afef", False)
            self._emit_chat_message(f"[System: Starting robust autonomous coding for '{self._original_query[:30]}...']")
            self._event_bus.uiInputBarBusyStateChanged.emit(True)

            # Create ENHANCED planning prompt for multi-file projects
            planning_prompt = self._create_enhanced_planning_prompt()
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

    def _create_enhanced_planning_prompt(self) -> str:
        """FIXED: Create enhanced planning prompt that forces correct format."""
        task_guidance = self._get_task_specific_guidance()

        return f"""You are an expert software architect. You MUST create a plan in the EXACT format specified below.

REQUEST: {self._original_query}
PROJECT DIRECTORY: {self._project_context['project_dir']}
TASK TYPE: {self._project_context.get('task_type', 'general')}

{task_guidance}

⚠️  CRITICAL FORMAT REQUIREMENTS ⚠️
You MUST respond in this EXACT format. Do NOT deviate from this structure:

## Architecture Overview
[Brief 2-3 sentence description of the approach and overall structure]

## File Dependencies & Generation Order
[Explain the dependency relationships and why files should be generated in a specific order]

## Files Required
FILES_LIST: ['file1.py', 'file2.py', 'file3.py']
GENERATION_ORDER: ['file1.py', 'file2.py', 'file3.py']

## Implementation Details

### file1.py
PURPOSE: [What this file does and why it's generated first]
DEPENDENCIES: []
DEPENDENTS: ['file2.py', 'file3.py']
PRIORITY: 1
REQUIREMENTS:
- [Specific requirement 1]
- [Specific requirement 2]
- [Include all necessary imports]
- [Define clear interfaces for other files to use]

### file2.py
PURPOSE: [What this file does]
DEPENDENCIES: ['file1.py']
DEPENDENTS: ['file3.py']
PRIORITY: 2
REQUIREMENTS:
- [Specific requirement 1]
- [Must import from file1.py]
- [Specific requirement 2]

### file3.py
PURPOSE: [What this file does]
DEPENDENCIES: ['file1.py', 'file2.py']
DEPENDENTS: []
PRIORITY: 3
REQUIREMENTS:
- [Specific requirement 1]
- [Must import from file1.py and file2.py]
- [Specific requirement 2]

## Error Handling Strategy
[How errors should be handled consistently across all files]

## Testing Strategy
[How the generated code should be testable]

🚨 MANDATORY FORMAT RULES 🚨
1. FILES_LIST must be a valid Python list: ['file1.py', 'file2.py']
2. GENERATION_ORDER must be a valid Python list: ['file1.py', 'file2.py']
3. Each file MUST have a ### section with PURPOSE, DEPENDENCIES, DEPENDENTS, PRIORITY, REQUIREMENTS
4. DEPENDENCIES must be a valid Python list: [] or ['file1.py']
5. DEPENDENTS must be a valid Python list: [] or ['file2.py']
6. PRIORITY must be a number: 1, 2, 3, etc.
7. Use EXACTLY the section headers shown above (##, ###)
8. Do NOT use formats like "FILES_TO_MODIFY:", "---CODER_INSTRUCTIONS_START---", etc.

EXAMPLE for a single file project:

## Architecture Overview
Simple single-file Python script that organizes files by type.

## File Dependencies & Generation Order
Single file with no dependencies - can be generated immediately.

## Files Required
FILES_LIST: ['file_organizer.py']
GENERATION_ORDER: ['file_organizer.py']

## Implementation Details

### file_organizer.py
PURPOSE: Main script that organizes files in a directory by moving them into categorized subdirectories
DEPENDENCIES: []
DEPENDENTS: []
PRIORITY: 1
REQUIREMENTS:
- Include command-line argument parsing with argparse
- Create get_file_category() function to determine file type
- Create organize_files() function to move files to appropriate subdirectories
- Include comprehensive error handling for file operations
- Add detailed docstrings and type hints throughout
- Support common file extensions (images, documents, videos, etc.)
- Create subdirectories automatically if they don't exist
- Skip the script itself when organizing files
- Print progress messages during file organization

CRITICAL: Follow this format EXACTLY. Your response will be parsed by code that expects these exact section headers and list formats."""

    def _get_task_specific_guidance(self) -> str:
        """Enhanced guidance based on detected task type."""
        task_type = self._project_context.get('task_type', 'general')

        guidance_map = {
            'api': """
For API projects, follow this structure:
- config.py (Priority 1): Configuration and constants
- models.py (Priority 2): Data models and schemas  
- database.py (Priority 3): Database connection and utilities
- handlers.py (Priority 4): Request handlers and business logic
- main.py (Priority 5): Application entry point and routing
Focus on RESTful APIs with proper validation, error handling, and response formatting.""",

            'data_processing': """
For data processing projects, follow this structure:
- config.py (Priority 1): Configuration and file paths
- utils.py (Priority 2): Common utilities and helpers
- validators.py (Priority 3): Data validation functions
- processors.py (Priority 4): Core processing logic
- main.py (Priority 5): Pipeline orchestration
Focus on efficient data handling, validation, and error recovery.""",

            'ui': """
For UI projects, follow this structure:
- constants.py (Priority 1): UI constants and themes
- models.py (Priority 2): Data models for UI state
- widgets.py (Priority 3): Reusable UI components
- dialogs.py (Priority 4): Dialog windows and forms
- main_window.py (Priority 5): Main application window
Focus on clean UI design with proper event handling.""",

            'utility': """
For utility projects, follow this structure:
- constants.py (Priority 1): Global constants
- exceptions.py (Priority 2): Custom exception classes
- utils.py (Priority 3): Core utility functions
- validators.py (Priority 4): Input validation functions
- main.py (Priority 5): CLI interface or main functionality
Focus on reusable, well-documented functions.""",

            'general': """
For general projects, follow this structure:
- config.py (Priority 1): Configuration and settings
- models.py (Priority 2): Data models and classes
- utils.py (Priority 3): Utility functions
- core.py (Priority 4): Main business logic
- main.py (Priority 5): Application entry point
Focus on clean, maintainable code with clear separation of concerns."""
        }

        return guidance_map.get(task_type, guidance_map['general'])

    @Slot(str, object, dict)
    def _handle_llm_completion(self, request_id: str, message: ChatMessage, metadata: dict):
        """Handle LLM response completion with robust multi-file support."""
        purpose = metadata.get("purpose")
        sequence_id = metadata.get("sequence_id")

        if sequence_id != self._sequence_id:
            return  # Not our sequence

        if purpose == "autonomous_planning" and request_id == self._planning_request_id:
            self._handle_planning_complete(message.text)
        elif purpose == "autonomous_coding" and request_id == self._current_generation_request_id:
            self._handle_code_generation_complete(request_id, message.text, metadata)

    def _handle_planning_complete(self, plan_text: str):
        """Handle completion of planning phase with dependency analysis."""
        logger.info("Planning phase completed, analyzing dependencies...")
        self._plan_text = plan_text
        self._planning_request_id = None
        self._current_phase = SequencePhase.DEPENDENCY_ANALYSIS

        try:
            # Parse the plan into file tasks
            self._file_tasks = self._parse_enhanced_plan_response(plan_text)

            if not self._file_tasks:
                self._handle_sequence_error("No files specified in plan")
                return

            # Build dependency graph and generation batches
            self._build_generation_batches()

            logger.info(
                f"Plan parsed successfully: {len(self._file_tasks)} files in {len(self._generation_batches)} batches")
            self._emit_chat_message(
                f"[System: Plan created for {len(self._file_tasks)} files in {len(self._generation_batches)} generation batches]")

            # Move to code generation phase
            self._current_phase = SequencePhase.CODE_GENERATION
            self._start_sequential_code_generation()

        except Exception as e:
            logger.error(f"Failed to parse plan or build dependencies: {e}", exc_info=True)
            self._handle_sequence_error(f"Plan analysis failed: {e}")

    def _parse_enhanced_plan_response(self, plan_text: str) -> List[FileGenerationTask]:
        """Parse the enhanced plan response with dependency information."""
        tasks = []

        # Extract files list
        files_match = re.search(r'FILES_LIST:\s*(\[.*?\])', plan_text, re.DOTALL)
        if not files_match:
            raise ValueError("FILES_LIST not found in plan")

        # Extract generation order
        order_match = re.search(r'GENERATION_ORDER:\s*(\[.*?\])', plan_text, re.DOTALL)

        try:
            files_list = eval(files_match.group(1))
            generation_order = eval(order_match.group(1)) if order_match else files_list

            if not isinstance(files_list, list) or not isinstance(generation_order, list):
                raise ValueError("FILES_LIST or GENERATION_ORDER is not a valid list")
        except Exception as e:
            raise ValueError(f"Failed to parse file lists: {e}")

        # Extract implementation details for each file
        for filename in files_list:
            section_pattern = rf'### {re.escape(filename)}\s*\n(.*?)(?=\n### |\n## |\Z)'
            section_match = re.search(section_pattern, plan_text, re.DOTALL)

            if section_match:
                section_content = section_match.group(1).strip()

                # Extract dependencies, dependents, and priority
                dependencies = self._extract_dependencies_from_section(section_content)
                dependents = self._extract_dependents_from_section(section_content)
                priority = self._extract_priority_from_section(section_content)
                instructions = self._extract_enhanced_file_instructions(section_content, filename)

            else:
                # Fallback
                dependencies = []
                dependents = []
                priority = generation_order.index(filename) + 1 if filename in generation_order else 999
                instructions = f"Implement {filename} according to the requirements in the plan."

            task = FileGenerationTask(
                filename=filename,
                instructions=instructions,
                task_type=self._detect_file_task_type(filename, instructions),
                dependencies=dependencies,
                dependents=dependents,
                generation_order=priority
            )
            tasks.append(task)

        # Sort by generation order
        tasks.sort(key=lambda t: t.generation_order)

        return tasks

    def _extract_dependencies_from_section(self, section_content: str) -> List[str]:
        """Extract dependencies from a file section."""
        deps_match = re.search(r'DEPENDENCIES:\s*(\[.*?\])', section_content, re.DOTALL)
        if deps_match:
            try:
                return eval(deps_match.group(1))
            except:
                pass
        return []

    def _extract_dependents_from_section(self, section_content: str) -> List[str]:
        """Extract dependents from a file section."""
        deps_match = re.search(r'DEPENDENTS:\s*(\[.*?\])', section_content, re.DOTALL)
        if deps_match:
            try:
                return eval(deps_match.group(1))
            except:
                pass
        return []

    def _extract_priority_from_section(self, section_content: str) -> int:
        """Extract priority from a file section."""
        priority_match = re.search(r'PRIORITY:\s*(\d+)', section_content)
        if priority_match:
            return int(priority_match.group(1))
        return 999

    def _extract_enhanced_file_instructions(self, section_content: str, filename: str) -> str:
        """Extract detailed instructions for a file from its plan section."""
        lines = section_content.split('\n')

        purpose = ""
        dependencies = []
        requirements = []

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('PURPOSE:'):
                purpose = line[8:].strip()
            elif line.startswith('DEPENDENCIES:'):
                deps_match = re.search(r'DEPENDENCIES:\s*(\[.*?\])', line)
                if deps_match:
                    try:
                        dependencies = eval(deps_match.group(1))
                    except:
                        pass
            elif line.startswith('REQUIREMENTS:'):
                current_section = 'requirements'
            elif line.startswith('- ') and current_section == 'requirements':
                requirements.append(line[2:].strip())

        # Build comprehensive instructions with context
        context_info = ""
        if dependencies:
            context_info = f"\n\nIMPORTANT: This file depends on: {', '.join(dependencies)}"
            context_info += "\nThese files have already been generated and are available for import."
            for dep in dependencies:
                if dep in self._generated_files_context:
                    context_info += f"\n\n{dep} contains:\n{self._get_file_summary(dep)}"

        instructions = f"""File: {filename}
Purpose: {purpose}

{context_info}

Detailed Requirements:
{chr(10).join(f'- {req}' for req in requirements)}

Implementation Guidelines:
- Follow Python best practices (PEP 8, type hints, comprehensive docstrings)
- Include robust error handling with specific exception types
- Add appropriate logging statements
- Validate all inputs and handle edge cases
- Write defensive code that fails gracefully
- Ensure code is production-ready and maintainable
- Include clear interfaces for files that will depend on this one

OUTPUT FORMAT: Respond with ONLY a Python code block:
```python
[YOUR COMPLETE, PRODUCTION-READY CODE HERE]
```

CRITICAL: The code must be complete, syntactically correct, and ready to run without modifications."""

        return instructions

    def _get_file_summary(self, filename: str) -> str:
        """Get a summary of an already generated file for context."""
        if filename not in self._generated_files_context:
            return "File not yet generated"

        content = self._generated_files_context[filename]

        # Extract key information: imports, classes, functions
        lines = content.split('\n')
        summary_lines = []

        for line in lines[:20]:  # First 20 lines usually contain imports and main definitions
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ', 'class ', 'def ', 'async def ')) or
                    (stripped.startswith('#') and len(stripped) > 5)):
                summary_lines.append(line)

        return '\n'.join(summary_lines)

    def _build_generation_batches(self):
        """Build generation batches based on dependencies."""
        self._generation_batches.clear()

        if self._generation_strategy == GenerationStrategy.SEQUENTIAL:
            # Generate one file at a time in dependency order
            for i, task in enumerate(self._file_tasks):
                batch = GenerationBatch(files=[task], batch_number=i + 1)
                self._generation_batches.append(batch)

        elif self._generation_strategy == GenerationStrategy.BATCHED:
            # Group files that can be generated in parallel (no dependencies between them)
            remaining_tasks = self._file_tasks.copy()
            batch_number = 1

            while remaining_tasks:
                current_batch_files = []
                files_to_remove = []

                for task in remaining_tasks:
                    # Check if all dependencies are already generated
                    deps_satisfied = all(
                        dep_file in [t.filename for batch in self._generation_batches for t in batch.files]
                        for dep_file in task.dependencies
                    )

                    if deps_satisfied and len(current_batch_files) < self._max_batch_size:
                        current_batch_files.append(task)
                        files_to_remove.append(task)

                if not current_batch_files:
                    # Prevent infinite loop - just take the first remaining task
                    current_batch_files = [remaining_tasks[0]]
                    files_to_remove = [remaining_tasks[0]]

                # Remove processed files
                for task in files_to_remove:
                    remaining_tasks.remove(task)

                batch = GenerationBatch(files=current_batch_files, batch_number=batch_number)
                self._generation_batches.append(batch)
                batch_number += 1

        else:  # PARALLEL - risky but fast
            # All files in one batch
            batch = GenerationBatch(files=self._file_tasks, batch_number=1)
            self._generation_batches.append(batch)

        logger.info(
            f"Created {len(self._generation_batches)} generation batches with strategy {self._generation_strategy.name}")

    def _start_sequential_code_generation(self):
        """Start sequential code generation process."""
        self._current_batch_index = 0
        self._emit_status(
            f"Starting code generation: {len(self._file_tasks)} files in {len(self._generation_batches)} batches",
            "#61afef", False)

        # Start the first batch
        asyncio.create_task(self._process_next_batch())

    async def _process_next_batch(self):
        """Process the next batch of files."""
        if self._current_batch_index >= len(self._generation_batches):
            # All batches completed
            self._handle_all_generation_complete()
            return

        current_batch = self._generation_batches[self._current_batch_index]
        logger.info(f"Processing batch {current_batch.batch_number} with {len(current_batch.files)} files")

        self._emit_status(
            f"Generating batch {current_batch.batch_number}/{len(self._generation_batches)} "
            f"({len(current_batch.files)} files)", "#61afef", False
        )

        # Generate files in this batch
        if self._generation_strategy == GenerationStrategy.SEQUENTIAL:
            # One at a time
            for task in current_batch.files:
                await self._generate_single_file_with_retry(task)
                await asyncio.sleep(self._generation_delay)  # Delay between generations
        else:
            # Multiple files in parallel (for batched strategy)
            await asyncio.gather(*[
                self._generate_single_file_with_retry(task)
                for task in current_batch.files
            ])

        # Move to next batch
        self._current_batch_index += 1
        await self._process_next_batch()

    async def _generate_single_file_with_retry(self, task: FileGenerationTask):
        """Generate a single file with retry logic."""
        while task.retry_count < task.max_retries:
            try:
                await self._generate_single_file(task)
                if task.generated_code:  # Success
                    break
            except Exception as e:
                logger.error(f"Error generating {task.filename} (attempt {task.retry_count + 1}): {e}")
                task.retry_count += 1
                task.error_message = str(e)

                if task.retry_count < task.max_retries:
                    logger.info(f"Retrying {task.filename} in {self._generation_delay}s...")
                    await asyncio.sleep(self._generation_delay)

    async def _generate_single_file(self, task: FileGenerationTask):
        """Generate a single file (async wrapper for sync process)."""
        task.request_id = f"code_{self._sequence_id}_{task.filename.replace('.', '_').replace('/', '_')}"
        self._current_generation_request_id = task.request_id

        # Update instructions with current context
        task.instructions = self._update_instructions_with_context(task)

        # Create coding prompt
        coding_prompt = self._create_coding_prompt(task)
        history = [ChatMessage(role=USER_ROLE, parts=[coding_prompt])]

        # Create a future to wait for completion
        self._generation_future = asyncio.Future()

        # Send to backend
        self._backend_coordinator.start_llm_streaming_task(
            request_id=task.request_id,
            target_backend_id=self._project_context['coder_backend'],
            history_to_send=history,
            is_modification_response_expected=True,
            options={"temperature": 0.1},
            request_metadata={
                "purpose": "autonomous_coding",
                "sequence_id": self._sequence_id,
                "filename": task.filename,
                "project_id": self._project_context.get('project_id'),
                "session_id": self._project_context.get('session_id')
            }
        )

        # Wait for completion (with timeout)
        try:
            await asyncio.wait_for(self._generation_future, timeout=120.0)  # 2 minute timeout per file
        except asyncio.TimeoutError:
            raise Exception(f"Generation timeout for {task.filename}")

    def _update_instructions_with_context(self, task: FileGenerationTask) -> str:
        """Update task instructions with current project context."""
        base_instructions = task.instructions

        # Add context about already generated files
        if self._generated_files_context:
            context_section = "\n\nPROJECT CONTEXT - Already Generated Files:\n"
            for filename, content in self._generated_files_context.items():
                context_section += f"\n{filename}:\n{self._get_file_summary(filename)}\n"

            base_instructions = base_instructions.replace(
                "Implementation Guidelines:",
                f"{context_section}\nImplementation Guidelines:"
            )

        return base_instructions

    def _create_coding_prompt(self, task: FileGenerationTask) -> str:
        """Create a focused coding prompt for a specific file."""
        return f"""You are a senior Python developer creating production-ready code. Generate complete, working code for:

FILE: {task.filename}
TASK TYPE: {task.task_type}
GENERATION ORDER: {task.generation_order} of {len(self._file_tasks)}

INSTRUCTIONS:
{task.instructions}

CRITICAL REQUIREMENTS:
1. Respond with ONLY a single Python code block
2. Use this EXACT format: ```python\\n[CODE]\\n```
3. NO explanatory text outside the code block
4. Include ALL necessary imports at the top
5. Add comprehensive docstrings (Google style)
6. Include complete type hints for all functions/methods
7. Implement robust error handling with try/except blocks
8. Add appropriate logging statements
9. Validate all inputs with clear error messages
10. Follow PEP 8 style guidelines strictly
11. Include constants at module level (no hardcoded values)
12. Write defensive code that handles edge cases gracefully

The code must be complete, syntactically correct, and ready to run in production without any modifications."""

    def _handle_code_generation_complete(self, request_id: str, raw_response: str, metadata: dict):
        """Handle completion of code generation for a file."""
        filename = metadata.get("filename", "unknown")

        # Find the task
        task = None
        for file_task in self._file_tasks:
            if file_task.request_id == request_id:
                task = file_task
                break

        if not task:
            logger.warning(f"No task found for request {request_id}")
            if hasattr(self, '_generation_future'):
                self._generation_future.set_exception(Exception("Task not found"))
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
            # Write to disk and add to context
            self._write_file_to_disk(task.filename, extracted_code)
            self._generated_files_context[task.filename] = extracted_code

            # Send to code viewer
            self._event_bus.modificationFileReadyForDisplay.emit(task.filename, extracted_code)

            logger.info(f"Successfully generated {task.filename} (Quality: {quality.name})")

            # Emit progress update
            completed_count = len(self._generated_files_context)
            self._emit_chat_message(f"[System: Generated {task.filename} ({completed_count}/{len(self._file_tasks)})]")

        else:
            task.error_message = f"Code extraction failed: {', '.join(notes)}"
            logger.warning(f"Failed to generate {task.filename}: {task.error_message}")

        # Signal completion
        if hasattr(self, '_generation_future'):
            self._generation_future.set_result(task)

    def _write_file_to_disk(self, filename: str, content: str):
        """Write generated file to disk with proper error handling."""
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
            raise

    def _handle_all_generation_complete(self):
        """Handle completion of all code generation tasks."""
        logger.info("All code generation tasks completed")

        # Count results
        successful = [t for t in self._file_tasks if t.generated_code and not t.error_message]
        failed = [t for t in self._file_tasks if t.error_message or not t.generated_code]

        # Create comprehensive summary
        summary_parts = [
            f"[System: Multi-file generation completed for '{self._original_query[:50]}...']",
            f"Successfully generated: {len(successful)}/{len(self._file_tasks)} files",
        ]

        if successful:
            files_list = ", ".join(f"`{t.filename}`" for t in successful)
            summary_parts.append(f"Generated files: {files_list}")

        if failed:
            failed_list = ", ".join(f"`{t.filename}` ({t.error_message or 'Unknown error'})" for t in failed)
            summary_parts.append(f"Failed files: {failed_list}")

        # Add quality summary
        quality_counts = {}
        for task in successful:
            if task.code_quality:
                quality_counts[task.code_quality.name] = quality_counts.get(task.code_quality.name, 0) + 1

        if quality_counts:
            quality_summary = ", ".join(f"{count} {quality}" for quality, count in quality_counts.items())
            summary_parts.append(f"Code quality: {quality_summary}")

        self._emit_chat_message(" | ".join(summary_parts))

        # Emit final status
        if len(successful) == len(self._file_tasks):
            color = "#98c379"  # All successful
            status = f"✅ Generated all {len(self._file_tasks)} files successfully!"
        elif successful:
            color = "#e5c07b"  # Partial success
            status = f"⚠️ Generated {len(successful)}/{len(self._file_tasks)} files"
        else:
            color = "#FF6B6B"  # All failed
            status = f"❌ Failed to generate any files"

        self._emit_status(status, color, False)

        # Log detailed results
        self._log_comm("MULTI_FILE_COMPLETE", f"Generated {len(successful)}/{len(self._file_tasks)} files")

        # Reset state
        self._reset_sequence()

    @Slot(str, str)
    def _handle_llm_error(self, request_id: str, error_message: str):
        """Handle LLM errors with proper retry logic."""
        if request_id == self._planning_request_id:
            self._handle_sequence_error(f"Planning failed: {error_message}")
        elif request_id == self._current_generation_request_id:
            # Signal error to waiting generation
            if hasattr(self, '_generation_future'):
                self._generation_future.set_exception(Exception(error_message))

    def _handle_sequence_error(self, error_message: str):
        """Handle sequence-level errors with cleanup."""
        logger.error(f"Sequence error: {error_message}")
        self._emit_chat_message(f"[System Error: {error_message}]", is_error=True)
        self._emit_status(f"Autonomous coding failed: {error_message}", "#FF6B6B", False)
        self._reset_sequence()

    def _reset_sequence(self):
        """Reset sequence state and cleanup."""
        self._current_phase = SequencePhase.IDLE
        self._sequence_id = None
        self._planning_request_id = None
        self._current_generation_request_id = None
        self._generation_batches.clear()
        self._current_batch_index = 0
        self._generated_files_context.clear()
        self._original_query = None
        self._project_context.clear()
        self._plan_text = None
        self._file_tasks.clear()

        self._event_bus.uiInputBarBusyStateChanged.emit(False)
        logger.info("Sequence state reset and cleaned up")

    # Helper methods
    def _detect_file_task_type(self, filename: str, instructions: str) -> str:
        """Detect task type for a specific file."""
        filename_lower = filename.lower()
        instructions_lower = instructions.lower()

        patterns = {
            'api': ['api', 'server', 'endpoint', 'route', 'handler', 'fastapi', 'flask'],
            'data_processing': ['data', 'process', 'parse', 'etl', 'csv', 'json', 'pandas'],
            'ui': ['ui', 'window', 'dialog', 'widget', 'gui', 'interface', 'qt'],
            'utility': ['util', 'helper', 'tool', 'lib', 'common', 'config'],
            'model': ['model', 'schema', 'entity', 'dto'],
            'database': ['db', 'database', 'sql', 'orm', 'migration']
        }

        for task_type, keywords in patterns.items():
            if any(keyword in filename_lower for keyword in keywords):
                return task_type
            if any(keyword in instructions_lower for keyword in keywords):
                return task_type

        return 'general'

    def _log_comm(self, prefix: str, message: str):
        """Log communication."""
        if self._llm_comm_logger:
            self._llm_comm_logger.log_message(f"PACC_V2:{prefix}", message)

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

    # Configuration methods
    def set_generation_strategy(self, strategy: GenerationStrategy):
        """Set the generation strategy (sequential, batched, or parallel)."""
        if self._current_phase == SequencePhase.IDLE:
            self._generation_strategy = strategy
            logger.info(f"Generation strategy set to: {strategy.name}")
        else:
            logger.warning("Cannot change generation strategy while sequence is active")

    def set_max_batch_size(self, size: int):
        """Set maximum batch size for batched generation."""
        if 1 <= size <= 10:
            self._max_batch_size = size
            logger.info(f"Max batch size set to: {size}")

    def set_generation_delay(self, delay: float):
        """Set delay between generations to avoid overwhelming LLM."""
        if 0.5 <= delay <= 10.0:
            self._generation_delay = delay
            logger.info(f"Generation delay set to: {delay}s")
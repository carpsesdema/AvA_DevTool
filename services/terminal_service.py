# services/terminal_service.py
import asyncio
import logging
import os
import subprocess
import sys  # For sys.stdout.encoding
import time
import uuid
from typing import Optional, List, Set, Dict, Any  # Keep Any if needed for future extensions
from pathlib import Path

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from utils import constants  # Assuming constants might be used later
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in TerminalService: {e}", exc_info=True)
    # Define fallbacks if necessary for the script to be parsable, though it will fail at runtime
    EventBus = type("EventBus", (object,), {"get_instance": lambda: type("DummyBus", (object,), {
        "terminalCommandRequested": type("Signal", (object,), {"connect": lambda x: None})(),
        "terminalCommandStarted": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandOutput": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandCompleted": type("Signal", (object,), {"emit": lambda *args: None})(),
        "terminalCommandError": type("Signal", (object,), {"emit": lambda *args: None})(),
    })()})
    constants = type("constants", (object,), {})
    raise  # Re-raise after attempting to define fallbacks for parsing

logger = logging.getLogger(__name__)


class TerminalService(QObject):
    """
    Service for executing terminal commands safely and streaming output.
    Integrates with the EventBus to emit command results.
    """

    # Safe commands that can be executed without user confirmation
    SAFE_COMMANDS = {
        # Python tools
        'python', 'python3', 'pip', 'pip3', 'pytest', 'black', 'mypy', 'flake8', 'isort',
        # Node/JS tools
        'node', 'npm', 'npx', 'yarn',
        # General development tools
        'git', 'ls', 'dir', 'cat', 'type', 'echo', 'pwd', 'cd', 'mkdir', 'touch',
        # Linting and formatting
        'pylint', 'autopep8', 'bandit', 'safety', 'ruff',
        # Testing
        'coverage', 'tox', 'nox',
        # Build tools
        'make', 'cmake', 'ninja', 'gradle', 'mvn',
        # Other common dev utilities
        'grep', 'find', 'awk', 'sed', 'curl', 'wget', 'unzip', 'tar', 'gzip', 'bzip2',
        'py_compile',  # Explicitly add py_compile as safe
    }

    # Commands that should never be executed automatically
    FORBIDDEN_COMMANDS = {
        'rm', 'del', 'rmdir', 'rd', 'format', 'fdisk', 'sudo', 'su', 'chmod', 'chown',
        'kill', 'killall', 'pkill', 'taskkill', 'shutdown', 'reboot', 'halt', 'poweroff',
        'dd', 'mkfs', 'fsck', 'mount', 'umount', 'crontab', 'systemctl', 'service',
        # Add potentially dangerous network commands if not intended for general use
        'iptables', 'ufw',
    }

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = EventBus.get_instance()
        self._active_processes: Dict[str, asyncio.subprocess.Process] = {}  # Store asyncio.subprocess.Process
        self._connect_signals()
        logger.info("TerminalService initialized")

    def _connect_signals(self):
        """Connect to EventBus signals"""
        self._event_bus.terminalCommandRequested.connect(self._handle_command_request)

    @Slot(str, str, str)
    def _handle_command_request(self, command: str, working_directory: str, command_id: str):
        """Handle terminal command execution request"""
        logger.info(f"Terminal command requested: '{command}' in '{working_directory}' (ID: {command_id})")
        asyncio.create_task(self._execute_command_async(command, working_directory, command_id))

    async def _execute_command_async(self, command: str, working_directory: str, command_id: str):
        """Execute command asynchronously and stream output"""
        start_time = time.time()
        process: Optional[asyncio.subprocess.Process] = None  # Ensure process is defined

        try:
            if not self._is_command_safe(command):
                error_msg = f"Command execution denied for safety reasons: '{command.split(' ')[0]}'"
                logger.warning(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            if not os.path.isdir(working_directory):  # Ensure it's a directory
                error_msg = f"Working directory does not exist or is not a directory: {working_directory}"
                logger.error(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            self._event_bus.terminalCommandStarted.emit(command_id, command)

            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
                # text=True, # REMOVED: This was the cause of "text must be False" error
            )
            self._active_processes[command_id] = process

            async def stream_output(stream: Optional[asyncio.StreamReader], output_type: str):
                if stream is None:
                    logger.warning(f"Stream for {output_type} is None for command {command_id}. Cannot read.")
                    return
                try:
                    while True:
                        line_bytes = await stream.readline()
                        if not line_bytes:
                            break
                        try:
                            # Try decoding with common encodings, prioritizing UTF-8
                            line_str = line_bytes.decode('utf-8', errors='replace')
                        except UnicodeDecodeError:
                            try:  # Fallback to system's default console encoding
                                line_str = line_bytes.decode(sys.stdout.encoding or 'latin-1', errors='replace')
                            except Exception:  # Ultimate fallback
                                line_str = line_bytes.decode('latin-1', errors='replace')
                        self._event_bus.terminalCommandOutput.emit(command_id, output_type, line_str.rstrip('\n'))
                except asyncio.CancelledError:
                    logger.info(f"Output streaming for {output_type} (command {command_id}) cancelled.")
                except Exception as e_stream:
                    logger.error(f"Error streaming {output_type} for command {command_id}: {e_stream}", exc_info=True)
                    # Emit error to terminal output as well
                    self._event_bus.terminalCommandOutput.emit(command_id, "stderr", f"[Streaming Error: {e_stream}]")

            stdout_task = asyncio.create_task(stream_output(process.stdout, "stdout"))
            stderr_task = asyncio.create_task(stream_output(process.stderr, "stderr"))

            await asyncio.gather(stdout_task, stderr_task)

            exit_code = await process.wait()
            execution_time = time.time() - start_time

            self._event_bus.terminalCommandCompleted.emit(command_id, exit_code, execution_time)
            logger.info(
                f"Command '{command}' (ID: {command_id}) completed with exit code: {exit_code}, time: {execution_time:.2f}s")

        except FileNotFoundError:  # If the command itself is not found
            error_msg = f"Command not found: '{command.split(' ')[0]}'. Ensure it's in your system's PATH."
            logger.error(error_msg, exc_info=True)
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        except PermissionError:
            error_msg = f"Permission denied to execute command: '{command}'"
            logger.error(error_msg, exc_info=True)
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        except Exception as e:
            error_msg = f"Error executing command '{command}' (ID: {command_id}): {type(e).__name__} - {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._event_bus.terminalCommandError.emit(command_id, error_msg)
        finally:
            if command_id in self._active_processes:
                del self._active_processes[command_id]
            # Ensure process is cleaned up if it exists and an error occurred before wait()
            if process and process.returncode is None:
                try:
                    process.kill()  # Force kill if still running after error
                    await process.wait()
                except Exception as e_kill:
                    logger.warning(f"Error trying to kill process {command_id} during cleanup: {e_kill}")

    def _is_command_safe(self, command: str) -> bool:
        if not command or not command.strip():
            return False

        parts = command.strip().split()
        base_command = parts[0].lower()
        base_command_path = Path(base_command)

        # Use the name of the executable, not the full path
        base_command_name = base_command_path.name

        # Check against forbidden commands first
        if base_command_name in self.FORBIDDEN_COMMANDS:
            logger.warning(f"Command '{base_command_name}' is in FORBIDDEN_COMMANDS.")
            return False

        # Check if it's a direct safe command
        if base_command_name in self.SAFE_COMMANDS:
            return True

        # Check for module execution like 'python -m module_name'
        if base_command_name in ('python', 'python3') and len(parts) > 2 and parts[1] == '-m':
            module_name = parts[2].lower()
            # Allow if the module itself is considered safe (e.g., 'py_compile', 'pip')
            if module_name in self.SAFE_COMMANDS:
                return True
            logger.warning(f"Python module '{module_name}' not in explicit SAFE_COMMANDS for module execution.")
            # Add more specific safe modules here if needed, e.g., 'pytest', 'black'
            # return module_name in {'py_compile', 'pip', 'pytest', 'black', 'mypy', 'flake8', 'isort', 'venv'}
            return False  # Default to not safe for unknown modules

        # Check for pip install (usually safe but be mindful of package sources)
        if base_command_name in ('pip', 'pip3') and len(parts) > 1 and parts[1].lower() == 'install':
            return True  # Generally allow pip install

        logger.warning(
            f"Command '{base_command_name}' not explicitly in SAFE_COMMANDS and doesn't match allowed patterns.")
        return False  # Default to not safe

    def execute_command(self, command: str, working_directory: Optional[str] = None) -> str:
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        work_dir = working_directory or os.getcwd()
        self._event_bus.terminalCommandRequested.emit(command, work_dir, command_id)
        return command_id

    def cancel_command(self, command_id: str) -> bool:
        if command_id in self._active_processes:
            process = self._active_processes[command_id]
            try:
                if process.returncode is None:  # Check if process is still running
                    logger.info(f"Attempting to terminate process for command ID: {command_id} (PID: {process.pid})")
                    process.terminate()  # Send SIGTERM
                    # Optionally, add a timeout and then process.kill() if terminate isn't enough
                # del self._active_processes[command_id] # Remove from active_processes after confirmation or in finally block
                return True
            except ProcessLookupError:  # Process already exited
                logger.warning(f"Process for command {command_id} already exited before explicit cancellation.")
                if command_id in self._active_processes: del self._active_processes[command_id]
                return True
            except Exception as e:
                logger.error(f"Error cancelling command {command_id}: {e}", exc_info=True)
                return False
        logger.info(f"Command ID {command_id} not found in active processes for cancellation.")
        return False

    def get_active_commands(self) -> List[str]:
        return list(self._active_processes.keys())

    def is_command_running(self, command_id: str) -> bool:
        process = self._active_processes.get(command_id)
        return process is not None and process.returncode is None
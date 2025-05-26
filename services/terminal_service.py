# services/terminal_service.py
import asyncio
import logging
import os
import subprocess
import time
import uuid
from typing import Optional, List, Set, Dict, Any
from pathlib import Path

from PySide6.QtCore import QObject, Slot

try:
    from core.event_bus import EventBus
    from utils import constants
except ImportError as e:
    logging.getLogger(__name__).critical(f"Critical import error in TerminalService: {e}", exc_info=True)
    raise

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
        'pylint', 'autopep8', 'bandit', 'safety',
        # Testing
        'coverage', 'tox', 'nox'
    }

    # Commands that should never be executed automatically
    FORBIDDEN_COMMANDS = {
        'rm', 'del', 'rmdir', 'rd', 'format', 'fdisk', 'sudo', 'su', 'chmod', 'chown',
        'kill', 'killall', 'pkill', 'taskkill', 'shutdown', 'reboot', 'halt', 'poweroff',
        'dd', 'mkfs', 'fsck', 'mount', 'umount', 'crontab', 'systemctl', 'service'
    }

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._event_bus = EventBus.get_instance()
        self._active_processes: Dict[str, subprocess.Popen] = {}
        self._connect_signals()
        logger.info("TerminalService initialized")

    def _connect_signals(self):
        """Connect to EventBus signals"""
        self._event_bus.terminalCommandRequested.connect(self._handle_command_request)

    @Slot(str, str, str)
    def _handle_command_request(self, command: str, working_directory: str, command_id: str):
        """Handle terminal command execution request"""
        logger.info(f"Terminal command requested: '{command}' in '{working_directory}' (ID: {command_id})")

        # Create async task for command execution
        asyncio.create_task(self._execute_command_async(command, working_directory, command_id))

    async def _execute_command_async(self, command: str, working_directory: str, command_id: str):
        """Execute command asynchronously and stream output"""
        start_time = time.time()

        try:
            # Validate command safety
            if not self._is_command_safe(command):
                error_msg = f"Command not allowed for safety: {command}"
                logger.warning(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            # Validate working directory
            if not os.path.exists(working_directory):
                error_msg = f"Working directory does not exist: {working_directory}"
                logger.error(error_msg)
                self._event_bus.terminalCommandError.emit(command_id, error_msg)
                return

            # Emit command started
            self._event_bus.terminalCommandStarted.emit(command_id, command)

            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            self._active_processes[command_id] = process

            # Stream stdout and stderr concurrently
            async def stream_output(stream, output_type):
                try:
                    while True:
                        line = await stream.readline()
                        if not line:
                            break
                        # Emit each line of output
                        self._event_bus.terminalCommandOutput.emit(command_id, output_type, line.rstrip('\n'))
                except Exception as e:
                    logger.error(f"Error streaming {output_type}: {e}")

            # Start streaming both stdout and stderr
            await asyncio.gather(
                stream_output(process.stdout, "stdout"),
                stream_output(process.stderr, "stderr")
            )

            # Wait for process to complete
            exit_code = await process.wait()
            execution_time = time.time() - start_time

            # Clean up
            if command_id in self._active_processes:
                del self._active_processes[command_id]

            # Emit completion
            self._event_bus.terminalCommandCompleted.emit(command_id, exit_code, execution_time)

            logger.info(f"Command completed: '{command}' (exit code: {exit_code}, time: {execution_time:.2f}s)")

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error executing command '{command}': {str(e)}"
            logger.error(error_msg)

            # Clean up
            if command_id in self._active_processes:
                del self._active_processes[command_id]

            self._event_bus.terminalCommandError.emit(command_id, error_msg)

    def _is_command_safe(self, command: str) -> bool:
        """Check if a command is safe to execute"""
        if not command or not command.strip():
            return False

        # Get the base command (first word)
        base_command = command.strip().split()[0].lower()

        # Remove common prefixes
        for prefix in ['python -m ', 'python3 -m ', 'pip install ', 'pip3 install ']:
            if command.lower().startswith(prefix):
                remaining = command[len(prefix):].strip().split()[0].lower()
                base_command = remaining
                break

        # Check against forbidden commands
        if base_command in self.FORBIDDEN_COMMANDS:
            return False

        # Check against safe commands
        if base_command in self.SAFE_COMMANDS:
            return True

        # Special cases for python modules
        if command.lower().startswith(('python -m ', 'python3 -m ')):
            return True

        # Special cases for pip installs
        if command.lower().startswith(('pip install ', 'pip3 install ')):
            return True

        # If not explicitly safe, reject
        logger.warning(f"Command not in safe list: {base_command}")
        return False

    def execute_command(self, command: str, working_directory: Optional[str] = None) -> str:
        """
        Public method to execute a command. Returns a command ID for tracking.
        """
        command_id = f"cmd_{uuid.uuid4().hex[:8]}"
        work_dir = working_directory or os.getcwd()

        # Emit the command request
        self._event_bus.terminalCommandRequested.emit(command, work_dir, command_id)

        return command_id

    def cancel_command(self, command_id: str) -> bool:
        """Cancel a running command"""
        if command_id in self._active_processes:
            try:
                process = self._active_processes[command_id]
                process.terminate()
                del self._active_processes[command_id]
                logger.info(f"Cancelled command: {command_id}")
                return True
            except Exception as e:
                logger.error(f"Error cancelling command {command_id}: {e}")
                return False
        return False

    def get_active_commands(self) -> List[str]:
        """Get list of currently running command IDs"""
        return list(self._active_processes.keys())

    def is_command_running(self, command_id: str) -> bool:
        """Check if a specific command is still running"""
        return command_id in self._active_processes
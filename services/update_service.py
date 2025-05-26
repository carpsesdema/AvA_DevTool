# services/update_service.py
import json
import logging
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from PySide6.QtCore import QObject, QThread, Signal, Slot

from utils import constants

logger = logging.getLogger(__name__)


class UpdateInfo:
    """Container for update information"""

    def __init__(self, data: Dict[str, Any]):
        self.version: str = data.get('version', '0.0.0')
        self.build_date: str = data.get('build_date', '')
        self.changelog: str = data.get('changelog', '')
        self.critical: bool = data.get('critical', False)
        self.min_version: str = data.get('min_version', '0.0.0')
        self.file_name: str = data.get('file_name', '')
        self.file_size: int = data.get('file_size', 0)
        self.download_url: str = data.get('download_url', '')

    @property
    def file_size_mb(self) -> float:
        return self.file_size / (1024 * 1024)

    def is_newer_than(self, current_version: str) -> bool:
        """Compare version strings (assumes semantic versioning X.Y.Z)"""
        try:
            current_parts = [int(x) for x in current_version.split('.')]
            update_parts = [int(x) for x in self.version.split('.')]

            # Pad shorter version with zeros
            max_len = max(len(current_parts), len(update_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            update_parts.extend([0] * (max_len - len(update_parts)))

            return update_parts > current_parts
        except ValueError:
            return False


class UpdateWorker(QThread):
    """Worker thread for downloading updates"""
    progress_updated = Signal(int)  # percentage
    status_updated = Signal(str)  # status message
    download_completed = Signal(str)  # file path
    download_failed = Signal(str)  # error message

    def __init__(self, update_info: UpdateInfo, download_dir: Path):
        super().__init__()
        self.update_info = update_info
        self.download_dir = download_dir
        self.should_stop = False

    def run(self):
        try:
            self.status_updated.emit("Downloading update...")
            self.download_dir.mkdir(exist_ok=True)

            file_path = self.download_dir / self.update_info.file_name

            response = requests.get(self.update_info.download_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.should_stop:
                        file_path.unlink(missing_ok=True)
                        return

                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if total_size > 0:
                            progress = int((downloaded_size / total_size) * 100)
                            self.progress_updated.emit(progress)

            self.status_updated.emit("Download completed!")
            self.download_completed.emit(str(file_path))

        except Exception as e:
            logger.error(f"Download failed: {e}")
            self.download_failed.emit(str(e))

    def stop(self):
        self.should_stop = True


class UpdateService(QObject):
    """Service for checking and applying application updates"""

    update_available = Signal(object)  # UpdateInfo
    no_update_available = Signal()
    update_check_failed = Signal(str)  # error message
    update_downloaded = Signal(str)  # file path
    update_download_failed = Signal(str)  # error message
    update_progress = Signal(int)  # download progress percentage
    update_status = Signal(str)  # status message

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)

        # GitHub repository info (should match build_and_deploy.py)
        self.github_repo = os.getenv('GITHUB_REPO', 'carpsesdema/AvA_DevTool')
        self.current_version = constants.APP_VERSION

        # Download directory
        self.update_dir = Path(constants.USER_DATA_DIR) / "updates"
        self.update_dir.mkdir(exist_ok=True)

        self.download_worker: Optional[UpdateWorker] = None

        logger.info(f"UpdateService initialized. Current version: {self.current_version}")

    def check_for_updates(self) -> None:
        """Check GitHub releases for newer versions"""
        try:
            logger.info("Checking for updates...")
            self.update_status.emit("Checking for updates...")

            # Get latest release info from GitHub
            api_url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            release_data = response.json()
            latest_version = release_data['tag_name'].lstrip('v')  # Remove 'v' prefix

            logger.info(f"Latest version on GitHub: {latest_version}")

            # Look for our executable in the assets
            assets = release_data.get('assets', [])
            app_asset = None

            # Match the naming pattern from build_and_deploy.py
            for asset in assets:
                asset_name = asset['name']
                if asset_name.startswith('AvA_DevTool_v') and (
                        asset_name.endswith('.exe') or
                        asset_name.endswith('_mac') or
                        asset_name.endswith('_linux') or
                        asset_name.endswith('.zip')
                ):
                    app_asset = asset
                    break

            if not app_asset:
                self.update_check_failed.emit("No compatible executable found in latest release")
                return

            # Create UpdateInfo object
            update_info = UpdateInfo({
                'version': latest_version,
                'build_date': release_data['published_at'],
                'changelog': release_data['body'],
                'critical': False,  # Could parse from changelog if needed
                'min_version': '0.1.0',
                'file_name': app_asset['name'],
                'file_size': app_asset['size'],
                'download_url': app_asset['browser_download_url']
            })

            if update_info.is_newer_than(self.current_version):
                logger.info(f"Update available: {latest_version}")
                self.update_available.emit(update_info)
            else:
                logger.info("No update available")
                self.no_update_available.emit()

        except requests.RequestException as e:
            error_msg = f"Network error checking for updates: {e}"
            logger.error(error_msg)
            self.update_check_failed.emit(error_msg)
        except Exception as e:
            error_msg = f"Error checking for updates: {e}"
            logger.error(error_msg)
            self.update_check_failed.emit(error_msg)

    def download_update(self, update_info: UpdateInfo) -> None:
        """Download the update file"""
        if self.download_worker and self.download_worker.isRunning():
            logger.warning("Download already in progress")
            return

        logger.info(f"Starting download of {update_info.file_name}")

        self.download_worker = UpdateWorker(update_info, self.update_dir)
        self.download_worker.progress_updated.connect(self.update_progress.emit)
        self.download_worker.status_updated.connect(self.update_status.emit)
        self.download_worker.download_completed.connect(self._on_download_completed)
        self.download_worker.download_failed.connect(self._on_download_failed)
        self.download_worker.start()

    @Slot(str)
    def _on_download_completed(self, file_path: str):
        logger.info(f"Update downloaded to: {file_path}")
        self.update_downloaded.emit(file_path)

    @Slot(str)
    def _on_download_failed(self, error_message: str):
        logger.error(f"Update download failed: {error_message}")
        self.update_download_failed.emit(error_message)

    def cancel_download(self) -> None:
        """Cancel ongoing download"""
        if self.download_worker and self.download_worker.isRunning():
            self.download_worker.stop()
            self.download_worker.wait(5000)  # Wait up to 5 seconds
            self.update_status.emit("Download cancelled")

    def apply_update(self, file_path: str) -> bool:
        """Apply the downloaded update"""
        try:
            update_file = Path(file_path)
            if not update_file.exists():
                logger.error(f"Update file not found: {file_path}")
                return False

            # Get current executable path
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                current_exe = Path(sys.executable)
            else:
                # Running from source - for testing
                logger.warning("Running from source, cannot apply update")
                return False

            # Create backup of current executable
            backup_path = current_exe.with_suffix('.backup')
            if backup_path.exists():
                backup_path.unlink()

            logger.info(f"Creating backup: {backup_path}")
            shutil.copy2(current_exe, backup_path)

            # Replace current executable with update
            logger.info(f"Applying update: {update_file} -> {current_exe}")
            shutil.copy2(update_file, current_exe)

            # Make executable on Unix systems
            if sys.platform != 'win32':
                os.chmod(current_exe, 0o755)

            logger.info("Update applied successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to apply update: {e}")
            return False

    def restart_application(self) -> None:
        """Restart the application after update"""
        try:
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                current_exe = sys.executable
                logger.info(f"Restarting application: {current_exe}")

                if sys.platform == 'win32':
                    subprocess.Popen([current_exe], creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                else:
                    subprocess.Popen([current_exe])

                # Exit current instance
                sys.exit(0)
            else:
                logger.warning("Cannot restart when running from source")

        except Exception as e:
            logger.error(f"Failed to restart application: {e}")

    def cleanup_old_files(self) -> None:
        """Clean up old update files and backups"""
        try:
            # Remove old update files
            for file_path in self.update_dir.glob("AvA_DevTool_v*"):
                if file_path.is_file():
                    file_path.unlink()
                    logger.debug(f"Cleaned up: {file_path}")

            # Remove old backup files
            if getattr(sys, 'frozen', False):
                current_exe = Path(sys.executable)
                backup_path = current_exe.with_suffix('.backup')
                if backup_path.exists():
                    backup_path.unlink()
                    logger.debug(f"Cleaned up backup: {backup_path}")

        except Exception as e:
            logger.warning(f"Error cleaning up old files: {e}")
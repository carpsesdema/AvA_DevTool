#!/usr/bin/env python3
"""
AvA Desktop App - Complete Build and Deployment Script
======================================================

This script handles:
1. Building the executable with PyInstaller (now in --onedir mode and zipping output)
2. Creating GitHub releases
3. Uploading the executable
4. Managing version updates

Usage:
    python build_and_deploy.py --version 1.0.5 --changelog "Bug fixes and improvements"
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import requests


class BuildError(Exception):
    """Custom exception for build-related errors"""
    pass


class DeploymentError(Exception):
    """Custom exception for deployment-related errors"""
    pass


class AvaBuilder:
    """Handles building and deploying the AvA Desktop application"""

    def __init__(self):
        self.project_root = Path.cwd()
        self.dist_dir = self.project_root / "dist"
        self.build_dir = self.project_root / "build"
        self.releases_dir = self.project_root / "releases"

        # GitHub configuration - UPDATE THESE!
        self.github_token = os.getenv('GITHUB_TOKEN2')
        # IMPORTANT: Update this to your AvA project's GitHub repository!
        self.github_repo = os.getenv('GITHUB_REPO2', 'carpsesdema/AvA_DevTool')

        # Application details
        self.app_name = "AvA_DevTool"  # Name for the executable
        self.main_script_path = "main.py"  # Relative path to your main application script
        self.version_file_path = Path("utils") / "constants.py"  # Relative path to file containing APP_VERSION

        print(f"🤖 AvA Desktop App Builder")
        print(f"📁 Project root: {self.project_root}")
        print(f"📦 Target repo: {self.github_repo}")
        print(f"🚀 App Name: {self.app_name}")
        print(f"📜 Main Script: {self.main_script_path}")
        print(f"🔢 Version File: {self.version_file_path}")

    def clean_build_dirs(self):
        """Clean previous build artifacts"""
        print("🧹 Cleaning build directories...")

        dirs_to_clean = [self.dist_dir, self.build_dir]
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"   Cleaned: {dir_path}")

        self.releases_dir.mkdir(exist_ok=True)
        print("✅ Build directories cleaned")

    def update_version_in_code(self, version: str):
        """Update the version number in the specified version file"""
        print(f"📝 Updating version to {version} in {self.version_file_path}...")

        version_file = self.project_root / self.version_file_path
        if not version_file.exists():
            raise BuildError(f"Version file not found: {version_file}")

        import re
        content = version_file.read_text(encoding='utf-8')

        # Target: APP_VERSION = "X.Y.Z"
        pattern = r'APP_VERSION\s*=\s*"[^"]*"'
        replacement = f'APP_VERSION = "{version}"'

        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            version_file.write_text(content, encoding='utf-8')
            print(f"✅ Version updated to {version} in {self.version_file_path}")
        else:
            print(f"⚠️  Warning: Could not find APP_VERSION in {self.version_file_path}")
            print("   This might be okay if version is defined elsewhere or format differs.")
            print(f"   Expected format: APP_VERSION = \"X.Y.Z\"")

    def build_executable(self, version: str) -> Path:
        """Build the executable using PyInstaller (now in --onedir mode and zips output)"""
        print("🔨 Building executable (one-dir mode)...")

        # This name will be used for the output folder in dist_dir and the final zip file base name
        base_name_with_version = f"{self.app_name}_v{version}"

        main_script_full_path = self.project_root / self.main_script_path
        if not main_script_full_path.exists():
            raise BuildError(f"Main script not found: {main_script_full_path}")

        try:
            import PyInstaller
        except ImportError:
            raise BuildError("PyInstaller is not installed. Run: pip install pyinstaller")

        cmd = [
            sys.executable, "-m", "PyInstaller",
            # "--onefile", # AVA_ASSISTANT_MODIFIED: Removed --onefile for directory-based build
            "--windowed",  # Keep for GUI apps
            "--name", base_name_with_version,  # PyInstaller will create a folder with this name
            "--distpath", str(self.dist_dir),
            "--workpath", str(self.build_dir),
            "--specpath", str(self.build_dir),
            "--clean",
            f"--icon={self.project_root / 'assets' / 'Synchat.ico'}",
        ]

        data_files_to_add = [
            ("assets", "assets"),
            ("ui", "ui"),
            ("backends", "backends"),
        ]
        for src, dest in data_files_to_add:
            src_path = self.project_root / src
            if src_path.exists():
                cmd.extend(["--add-data", f"{src_path}{os.pathsep}{dest}"])
            else:
                print(f"   Warning: Data source path not found, skipping: {src_path}")

        hidden_imports = [
            "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtSvg", "PySide6.QtNetwork",
            "qasync",
            "google.generativeai",
            "dotenv",
            "pygments", "pygments.lexers", "pygments.formatters", "pygments.styles",
            "markdown", "html", "html.parser",
            "openai",
            "ollama",
            "qtawesome",
            "chromadb", "sqlite3",
            "pypdf",
            "langchain_text_splitters", "langchain",
            "sentence_transformers", "transformers", "torch", "torchvision", "torchaudio",
            "asyncio",
            "importlib_resources",
            "logging", "json", "uuid", "datetime", "re", "pathlib", "shutil", "argparse", "requests",
            "tiktoken",
        ]
        for imp in hidden_imports:
            cmd.extend(["--hidden-import", imp])

        collect_data_packages = [
            "qtawesome",
            "pygments",
            "sentence_transformers",
            "transformers",
            "chromadb",
            "tiktoken",
        ]
        for pkg_name in collect_data_packages:
            cmd.extend(["--collect-data", pkg_name])

        cmd.append(str(main_script_full_path))

        try:
            print(f"   Running PyInstaller: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

            # In --onedir mode, PyInstaller creates a folder in dist_dir
            # The folder name is base_name_with_version
            output_dir_in_dist = self.dist_dir / base_name_with_version
            if not output_dir_in_dist.is_dir():
                raise BuildError(f"Output directory not found after build at expected path: {output_dir_in_dist}")

            # The executable is inside this directory
            # exe_file_name = f"{base_name_with_version}.exe" if sys.platform == "win32" else base_name_with_version
            # No, PyInstaller names the executable inside the folder the same as the folder
            exe_inside_dir_name = base_name_with_version
            if sys.platform == "win32":
                exe_inside_dir_name += ".exe"
            elif sys.platform == "darwin" and (output_dir_in_dist / f"{base_name_with_version}.app").is_dir():
                # If it created an .app bundle inside the dir (less common for pure --onedir unless specified)
                exe_inside_dir_name += ".app"  # This would be the .app bundle itself.

            executable_path_in_output_dir = output_dir_in_dist / exe_inside_dir_name

            if not executable_path_in_output_dir.exists() and not (
                    sys.platform == "darwin" and executable_path_in_output_dir.is_dir()):  # .is_dir for .app
                # Try without extension for Linux/Mac bare executable if .app check failed
                if not sys.platform.startswith("win32") and (output_dir_in_dist / base_name_with_version).exists():
                    executable_path_in_output_dir = output_dir_in_dist / base_name_with_version
                else:
                    raise BuildError(f"Executable not found within output directory: {executable_path_in_output_dir}")

            print(f"   Build output directory: {output_dir_in_dist}")
            print(f"   Main executable found at: {executable_path_in_output_dir}")

            # Determine the name for the final zip file
            platform_suffix = ""
            if sys.platform == "win32":
                platform_suffix = "_windows"
            elif sys.platform == "darwin":
                platform_suffix = "_macos"
            elif sys.platform.startswith("linux"):
                platform_suffix = "_linux"

            zip_file_name = f"{base_name_with_version}{platform_suffix}.zip"
            final_zip_path = self.releases_dir / zip_file_name

            print(f"   Zipping output directory to: {final_zip_path}")
            shutil.make_archive(
                base_name=str(final_zip_path.with_suffix('')),  # make_archive adds .zip itself
                format='zip',
                root_dir=self.dist_dir,  # The directory containing the folder to zip
                base_dir=base_name_with_version  # The name of the folder to zip (relative to root_dir)
            )

            if not final_zip_path.exists():
                raise BuildError(f"Zipped output not found at: {final_zip_path}")

            file_size_mb = final_zip_path.stat().st_size / (1024 * 1024)
            print(f"✅ Build and zipping completed successfully!")
            print(f"   📦 Zipped Release: {final_zip_path}")
            print(f"   📏 Size: {file_size_mb:.1f} MB")
            return final_zip_path  # Return path to the zip file

        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller failed:")
            print(f"   Output: {e.stdout}")
            print(f"   Error: {e.stderr}")
            raise BuildError(f"PyInstaller failed: {e}")
        except Exception as e_build:
            print(f"❌ An unexpected error occurred during build: {e_build}")
            raise BuildError(f"Build error: {e_build}")

    def create_release_info(self, version: str, changelog: str, exe_path: Path) -> dict:
        """Create release information JSON (exe_path is now the path to the ZIP)"""
        release_info = {
            "version": version,
            "build_date": datetime.now().isoformat(),
            "changelog": changelog,
            "critical": False,
            "min_version": "0.1.0",
            "file_name": exe_path.name,  # This is now the zip file name
            "file_size": exe_path.stat().st_size,
            "download_url": f"https://github.com/{self.github_repo}/releases/download/v{version}/{exe_path.name}"
        }

        info_file = self.releases_dir / f"update_info_v{version}.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(release_info, f, indent=2)

        print(f"📋 Release info saved: {info_file}")
        return release_info

    def create_git_tag(self, version: str) -> bool:
        """Create and push git tag"""
        tag_name = f'v{version}'
        try:
            print(f"📝 Creating git tag {tag_name}...")
            tag_check_result = subprocess.run(['git', 'tag', '-l', tag_name], capture_output=True, text=True)
            if tag_name in tag_check_result.stdout.split():
                print(f"   Git tag {tag_name} already exists locally.")
            else:
                result_create = subprocess.run(['git', 'tag', tag_name], capture_output=True, text=True, check=False)
                if result_create.returncode != 0:
                    print(f"   Warning: Could not create tag {tag_name} locally: {result_create.stderr}")
                else:
                    print(f"   Git tag {tag_name} created locally.")

            print(f"   Pushing tag {tag_name} to origin...")
            result_push = subprocess.run(['git', 'push', 'origin', tag_name], capture_output=True, text=True,
                                         check=False)
            if result_push.returncode == 0:
                print(f"✅ Git tag {tag_name} pushed to origin.")
                return True
            elif "already exists" in result_push.stderr.lower() or "up-to-date" in result_push.stderr.lower():
                print(f"   Git tag {tag_name} likely already exists on origin or is up-to-date.")
                return True
            else:
                print(f"   Warning: Could not push tag {tag_name}: {result_push.stderr}")
                print(f"   You might need to push it manually: git push origin {tag_name}")
                return False

        except FileNotFoundError:
            print("   Warning: Git command not found. Skipping tag creation/push.")
            return True
        except Exception as e:
            print(f"   Warning: Git tag operation failed: {e}")
            return True

    def create_github_release(self, version: str, changelog: str, exe_path: Path) -> bool:
        """Create a GitHub release and upload the executable (now a ZIP file)"""
        if not self.github_token:
            print("⚠️  GITHUB_TOKEN not found in environment variables.")
            return False

        if not self.create_git_tag(version):
            print("   Skipping GitHub release creation due to tag push failure.")
            return False

        print("🚀 Creating GitHub release...")
        release_tag = f"v{version}"
        release_data = {
            "tag_name": release_tag,
            "target_commitish": "main",
            "name": f"{self.app_name} v{version}",
            "body": changelog,
            "draft": False,
            "prerelease": False
        }
        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": f"{self.app_name}-Builder"
        }

        try:
            check_url = f"https://api.github.com/repos/{self.github_repo}/releases/tags/{release_tag}"
            print(f"   Checking for existing release: {check_url}")
            check_response = requests.get(check_url, headers=headers, timeout=10)

            if check_response.status_code == 200:
                release_info = check_response.json()
                print(f"✅ GitHub release for tag {release_tag} already exists!")
                asset_exists = any(asset['name'] == exe_path.name for asset in release_info.get('assets', []))
                if asset_exists:
                    print(f"   📦 Asset {exe_path.name} already uploaded. Skipping upload.")
                    return True
                else:
                    print(f"   Asset {exe_path.name} not found in existing release. Attempting to upload...")
                    return self._upload_asset_to_release(release_info, exe_path, headers)

            elif check_response.status_code == 404:
                print(f"   Creating new release for {self.github_repo}...")
                create_url = f"https://api.github.com/repos/{self.github_repo}/releases"
                response = requests.post(create_url, headers=headers, json=release_data, timeout=30)
                if response.status_code == 201:
                    release_info = response.json()
                    print(f"✅ GitHub release created!")
                    return self._upload_asset_to_release(release_info, exe_path, headers)
                else:
                    print(
                        f"❌ Failed to create GitHub release: Status {response.status_code}, Response: {response.text}")
                    return False
            else:
                print(
                    f"❌ Error checking for existing release: Status {check_response.status_code}, Response: {check_response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ GitHub API request error: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during GitHub release: {e}")
            return False

    def _upload_asset_to_release(self, release_info: dict, exe_path: Path, headers: dict) -> bool:
        """Upload asset (now a ZIP file) to GitHub release"""
        print(f"📤 Uploading {exe_path.name} to GitHub release...")
        upload_url_template = release_info['upload_url']
        upload_url = upload_url_template.split('{?name,label}')[0] + f"?name={exe_path.name}"

        try:
            with open(exe_path, 'rb') as f_asset:
                asset_data = f_asset.read()
            upload_headers = headers.copy()
            # AVA_ASSISTANT_MODIFIED: Content-Type should be application/zip
            upload_headers['Content-Type'] = 'application/zip'
            upload_response = requests.post(upload_url, headers=upload_headers, data=asset_data,
                                            timeout=600)  # Increased timeout for larger zips

            if upload_response.status_code == 201:
                asset_info = upload_response.json()
                print("✅ Asset uploaded successfully!")
                print(f"   📦 Download URL: {asset_info['browser_download_url']}")
                return True
            else:
                print(f"❌ Failed to upload asset:")
                print(f"   Status: {upload_response.status_code}")
                print(f"   Response: {upload_response.text}")
                try:
                    print(f"   Details: {upload_response.json()}")
                except json.JSONDecodeError:
                    pass
                return False
        except requests.exceptions.RequestException as e_req:
            print(f"❌ Upload request error: {e_req}")
            return False
        except Exception as e_upload:
            print(f"❌ Upload error: {e_upload}")
            return False

    def build_and_deploy(self, version: str, changelog: str, deploy_to_github: bool = True) -> bool:
        """Complete build and deploy process"""
        print(f"🚀 Starting build and deploy process for AvA v{version}")
        print("=" * 70)
        try:
            self.clean_build_dirs()
            self.update_version_in_code(version)
            final_asset_path = self.build_executable(version)  # This now returns the path to the final ZIP
            self.create_release_info(version, changelog, final_asset_path)

            if deploy_to_github:
                success = self.create_github_release(version, changelog, final_asset_path)
                if success:
                    print("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
                    return True
                else:
                    print("\n⚠️  GitHub deployment failed, but release asset is ready.")
                    print(f"   📦 Release asset location: {final_asset_path}")
                    return False
            else:
                print("\n📦 Build completed. Release asset ready for manual deployment.")
                print(f"   📦 Location: {final_asset_path}")
                return True
        except BuildError as e_build:
            print(f"❌ BUILD FAILED: {e_build}")
            return False
        except DeploymentError as e_deploy:
            print(f"❌ DEPLOYMENT FAILED: {e_deploy}")
            return False
        except Exception as e_main:
            print(f"❌ AN UNEXPECTED ERROR OCCURRED: {e_main}")
            import traceback
            traceback.print_exc()
            return False


def validate_version_format(version: str) -> bool:
    import re
    return bool(re.fullmatch(r'^\d+\.\d+\.\d+$', version))


def main():
    parser = argparse.ArgumentParser(
        description='Build and deploy AvA Desktop App updates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_and_deploy.py --version 0.1.0 --changelog "Initial release with core features."
  python build_and_deploy.py --version 0.1.1 --changelog "Fixed critical bug in chat UI." --no-github

Environment Variables:
  GITHUB_TOKEN    Your GitHub personal access token (for creating releases).
  GITHUB_REPO     Your repository (e.g., your_username/ava-desktop-app).
        """
    )
    parser.add_argument('--version', required=True, help='Version number (e.g., 0.1.0)')
    parser.add_argument('--changelog', required=True, help='Changelog description for this version')
    parser.add_argument('--no-github', action='store_true', help='Skip GitHub release and upload')
    args = parser.parse_args()

    if not validate_version_format(args.version):
        print(f"❌ Invalid version format: '{args.version}'. Must be X.Y.Z (e.g., 0.1.0)")
        sys.exit(1)

    builder_check = AvaBuilder()
    if not (builder_check.project_root / builder_check.main_script_path).exists():
        print(f"❌ Main script '{builder_check.main_script_path}' not found.")
        sys.exit(1)
    if not (builder_check.project_root / builder_check.version_file_path).exists():
        print(f"❌ Version file '{builder_check.version_file_path}' not found.")
        sys.exit(1)
    del builder_check

    builder = AvaBuilder()
    success = builder.build_and_deploy(
        version=args.version,
        changelog=args.changelog,
        deploy_to_github=not args.no_github
    )

    if success:
        print("\n🎉 All tasks completed successfully!" if not args.no_github else "\n🎉 Build completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Process failed. Check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
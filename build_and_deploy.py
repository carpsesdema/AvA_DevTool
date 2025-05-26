#!/usr/bin/env python3
"""
AvA Desktop App - Complete Build and Deployment Script
======================================================

This script handles:
1. Building the executable with PyInstaller
2. Creating GitHub releases
3. Uploading the executable
4. Managing version updates

Usage:
    python build_and_deploy.py --version 1.0.1 --changelog "Bug fixes and improvements"
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
        self.github_token = os.getenv('GITHUB_TOKEN')
        # IMPORTANT: Update this to your AvA project's GitHub repository!
        self.github_repo = os.getenv('GITHUB_REPO', 'carpsesdema/AvA_DevTool')

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
        """Build the executable using PyInstaller"""
        print("🔨 Building executable...")

        exe_name_with_version = f"{self.app_name}_v{version}"

        main_script_full_path = self.project_root / self.main_script_path
        if not main_script_full_path.exists():
            raise BuildError(f"Main script not found: {main_script_full_path}")

        try:
            import PyInstaller
        except ImportError:
            raise BuildError("PyInstaller is not installed. Run: pip install pyinstaller")

        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",
            "--windowed",  # Use --noconsole on Windows, --windowed for macOS/Linux GUI
            # "--noconsole", # Use for Windows GUI apps to hide console, implied by --windowed sometimes
            "--name", exe_name_with_version,
            "--distpath", str(self.dist_dir),
            "--workpath", str(self.build_dir),
            "--specpath", str(self.build_dir),
            "--clean",
            # Add icon if you have one
            # f"--icon={self.project_root / 'assets' / 'ava_logo.ico'}", # Example
        ]

        # Add data files (assets, UI styles)
        # Syntax: --add-data "source:destination_in_bundle"
        # Destination is relative to the main script's location in bundle
        data_files_to_add = [
            ("assets", "assets"),  # Bundles the 'assets' folder into an 'assets' folder next to exe
            ("ui", "ui"),  # Bundles the 'ui' folder (for style.qss, bubble_style.qss)
        ]
        for src, dest in data_files_to_add:
            src_path = self.project_root / src
            if src_path.exists():
                cmd.extend(["--add-data", f"{src_path}{os.pathsep}{dest}"])
            else:
                print(f"   Warning: Data source path not found, skipping: {src_path}")

        # Hidden imports for PySide6 and other dependencies
        hidden_imports = [
            "PySide6.QtCore", "PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtSvg", "PySide6.QtNetwork",
            "qasync",
            "google.generativeai",
            "dotenv",
            "pygments", "pygments.lexers", "pygments.formatters", "pygments.styles",
            "markdown", "html", "html.parser",  # Markdown and its deps
            "openai",
            "ollama",
            "qtawesome",
            "chromadb", "sqlite3",  # ChromaDB often uses sqlite
            "pypdf",
            "langchain_text_splitters",
            "sentence_transformers", "transformers", "torch",  # Sentence Transformers can be heavy
            "asyncio",
            "importlib_resources",
            "logging", "json", "uuid", "datetime", "re", "pathlib", "shutil", "argparse", "requests",
            # Add any other modules PyInstaller might miss
        ]
        for imp in hidden_imports:
            cmd.extend(["--hidden-import", imp])

        # Collect data for specific packages
        # This helps PyInstaller find data files used by these libraries (e.g., fonts, models)
        collect_data_packages = [
            "qtawesome",
            "pygments",
            # "sentence_transformers", # This can be complex, try if needed
            # "chromadb", # Might also need this
        ]
        for pkg_name in collect_data_packages:
            cmd.extend(["--collect-data", pkg_name])

        cmd.append(str(main_script_full_path))

        try:
            print(f"   Running PyInstaller: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')

            # PyInstaller places the .exe inside a folder with the exe_name in dist_dir if not using --onedir
            # For --onefile, it's directly in dist_dir
            exe_path_in_dist = self.dist_dir / f"{exe_name_with_version}.exe"  # Adjust for macOS/Linux if needed
            if not exe_path_in_dist.exists():
                # On macOS/Linux, it might be just exe_name_with_version (no .exe)
                # or inside an app bundle on macOS. This part might need platform-specific logic.
                exe_path_in_dist_alt = self.dist_dir / exe_name_with_version
                if exe_path_in_dist_alt.exists():
                    exe_path_in_dist = exe_path_in_dist_alt
                else:  # Check for .app bundle on macOS
                    mac_app_bundle = self.dist_dir / f"{exe_name_with_version}.app"
                    if mac_app_bundle.is_dir():
                        # For .app bundles, you might want to zip it before moving/uploading
                        # For now, just acknowledge it
                        print(f"   Found macOS .app bundle: {mac_app_bundle}")
                        # This example focuses on .exe, adjust if .app is primary target
                        # For simplicity, we'll assume we're building an .exe or single executable
                        # If .app bundling is the main goal, the logic for final_path and upload needs adjustment.
                        # One strategy is to zip the .app bundle.
                        final_path_target_name = f"{self.app_name}_v{version}_macOS.zip"
                        final_path = self.releases_dir / final_path_target_name
                        shutil.make_archive(str(final_path.with_suffix('')), 'zip', root_dir=self.dist_dir,
                                            base_dir=f"{exe_name_with_version}.app")
                        print(f"   📦 macOS .app bundle zipped to: {final_path}")
                        file_size = final_path.stat().st_size / (1024 * 1024)  # MB
                        print(f"✅ Build completed successfully!")
                        print(f"   📏 Size: {file_size:.1f} MB")
                        return final_path

            if exe_path_in_dist.exists():
                final_path_target_name = f"{self.app_name}_v{version}.exe"  # Default to .exe
                if sys.platform == "darwin" and not exe_path_in_dist.name.endswith(
                        ".exe"):  # if it's a bare executable on mac
                    final_path_target_name = f"{self.app_name}_v{version}_mac"
                elif sys.platform.startswith("linux") and not exe_path_in_dist.name.endswith(
                        ".exe"):  # if it's a bare executable on linux
                    final_path_target_name = f"{self.app_name}_v{version}_linux"

                final_path = self.releases_dir / final_path_target_name
                shutil.move(str(exe_path_in_dist), str(final_path))

                file_size = final_path.stat().st_size / (1024 * 1024)  # MB
                print(f"✅ Build completed successfully!")
                print(f"   📦 Executable: {final_path}")
                print(f"   📏 Size: {file_size:.1f} MB")
                return final_path
            else:
                raise BuildError(f"Executable not found after build at expected path: {exe_path_in_dist} (or similar)")

        except subprocess.CalledProcessError as e:
            print(f"❌ PyInstaller failed:")
            print(f"   Output: {e.stdout}")
            print(f"   Error: {e.stderr}")
            raise BuildError(f"PyInstaller failed: {e}")
        except Exception as e_build:
            print(f"❌ An unexpected error occurred during build: {e_build}")
            raise BuildError(f"Build error: {e_build}")

    def create_release_info(self, version: str, changelog: str, exe_path: Path) -> dict:
        """Create release information JSON"""
        release_info = {
            "version": version,
            "build_date": datetime.now().isoformat(),
            "changelog": changelog,
            "critical": False,  # Set to True for critical updates
            "min_version": "0.1.0",  # Minimum supported version, update as needed
            "file_name": exe_path.name,
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

            # Check if tag already exists locally
            tag_check_result = subprocess.run(['git', 'tag', '-l', tag_name], capture_output=True, text=True)
            if tag_name in tag_check_result.stdout.split():
                print(f"   Git tag {tag_name} already exists locally.")
            else:
                # Create tag locally
                result_create = subprocess.run(['git', 'tag', tag_name], capture_output=True, text=True, check=False)
                if result_create.returncode != 0:
                    print(f"   Warning: Could not create tag {tag_name} locally: {result_create.stderr}")
                else:
                    print(f"   Git tag {tag_name} created locally.")

            # Push tag to GitHub
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
                return False  # Tag push is important for release linking

        except FileNotFoundError:
            print("   Warning: Git command not found. Skipping tag creation/push.")
            print("   Ensure Git is installed and in your PATH for automatic tagging.")
            return True  # Continue build, but tag won't be auto-created
        except Exception as e:
            print(f"   Warning: Git tag operation failed: {e}")
            return True  # Continue anyway

    def create_github_release(self, version: str, changelog: str, exe_path: Path) -> bool:
        """Create a GitHub release and upload the executable"""
        if not self.github_token:
            print("⚠️  GITHUB_TOKEN not found in environment variables.")
            print("   To set up automatic GitHub releases:")
            print("   1. Go to GitHub Settings > Developer settings > Personal access tokens (classic or fine-grained)")
            print("   2. Create a token with 'repo' (or 'contents:write' for releases) permissions.")
            print("   3. Set environment variable: GITHUB_TOKEN=your_token_here")
            print("")
            print("   For now, you can manually upload the executable to GitHub releases.")
            return False

        if not self.create_git_tag(version):
            print("   Skipping GitHub release creation due to tag push failure.")
            return False

        print("🚀 Creating GitHub release...")
        release_tag = f"v{version}"
        release_data = {
            "tag_name": release_tag,
            "target_commitish": "main",  # or "master", or a specific commit SHA
            "name": f"{self.app_name} v{version}",
            "body": changelog,
            "draft": False,
            "prerelease": False  # Set to True if it's a pre-release
        }

        headers = {
            "Authorization": f"Bearer {self.github_token}",  # Recommended: Bearer
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",  # Good practice
            "User-Agent": f"{self.app_name}-Builder"
        }

        try:
            # Check if release already exists
            check_url = f"https://api.github.com/repos/{self.github_repo}/releases/tags/{release_tag}"
            print(f"   Checking for existing release: {check_url}")
            check_response = requests.get(check_url, headers=headers, timeout=10)

            if check_response.status_code == 200:
                release_info = check_response.json()
                print(f"✅ GitHub release for tag {release_tag} already exists!")
                print(f"   🔗 URL: {release_info['html_url']}")
                # Check if asset already exists
                asset_exists = any(asset['name'] == exe_path.name for asset in release_info.get('assets', []))
                if asset_exists:
                    print(f"   📦 Asset {exe_path.name} already uploaded. Skipping upload.")
                    return True
                else:
                    print(f"   Asset {exe_path.name} not found in existing release. Attempting to upload...")
                    return self._upload_asset_to_release(release_info, exe_path, headers)

            elif check_response.status_code == 404:
                # Release does not exist, create it
                print(f"   Creating new release for {self.github_repo}...")
                create_url = f"https://api.github.com/repos/{self.github_repo}/releases"
                response = requests.post(create_url, headers=headers, json=release_data, timeout=30)

                if response.status_code == 201:
                    release_info = response.json()
                    print(f"✅ GitHub release created!")
                    print(f"   🔗 URL: {release_info['html_url']}")
                    return self._upload_asset_to_release(release_info, exe_path, headers)
                else:
                    print(f"❌ Failed to create GitHub release:")
                    print(f"   Status: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return False
            else:
                print(f"❌ Error checking for existing release:")
                print(f"   Status: {check_response.status_code}")
                print(f"   Response: {check_response.text}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ GitHub API request error: {e}")
            return False
        except Exception as e:
            print(f"❌ An unexpected error occurred during GitHub release: {e}")
            return False

    def _upload_asset_to_release(self, release_info: dict, exe_path: Path, headers: dict) -> bool:
        """Upload executable to GitHub release"""
        print(f"📤 Uploading {exe_path.name} to GitHub release...")

        upload_url_template = release_info[
            'upload_url']  # upload_url is a template: "https://uploads.github.com/repos/owner/repo/releases/id/assets{?name,label}"
        upload_url = upload_url_template.split('{?name,label}')[0] + f"?name={exe_path.name}"

        try:
            with open(exe_path, 'rb') as f_asset:
                asset_data = f_asset.read()

            upload_headers = headers.copy()
            upload_headers['Content-Type'] = 'application/octet-stream'  # For .exe, .zip etc.
            # For .app.zip, use application/zip

            upload_response = requests.post(
                upload_url,
                headers=upload_headers,
                data=asset_data,
                timeout=300  # 5 minutes for potentially large files
            )

            if upload_response.status_code == 201:
                asset_info = upload_response.json()
                print("✅ Executable uploaded successfully!")
                print(f"   📦 Download URL: {asset_info['browser_download_url']}")
                return True
            else:
                print(f"❌ Failed to upload executable:")
                print(f"   Status: {upload_response.status_code}")
                print(f"   Response: {upload_response.text}")
                # Try to get more details from response if JSON
                try:
                    error_details = upload_response.json()
                    print(f"   Details: {error_details}")
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
            exe_path = self.build_executable(version)  # This now returns the path to the final asset
            self.create_release_info(version, changelog, exe_path)

            if deploy_to_github:
                success = self.create_github_release(version, changelog, exe_path)
                if success:
                    print("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
                    print(f"   📱 Clients using an auto-updater will be notified of v{version} (if implemented).")
                    print(f"   📥 They can download from GitHub Releases.")
                    return True
                else:
                    print("\n⚠️  GitHub deployment failed, but executable is ready.")
                    print(f"   📦 Executable location: {exe_path}")
                    print("   You can manually upload it to GitHub releases.")
                    return False  # Indicate deployment part failed
            else:
                print("\n📦 Build completed. Executable ready for manual deployment.")
                print(f"   📦 Location: {exe_path}")
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
    """Validate version format (e.g., X.Y.Z)"""
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
                  Default can be set in the script if this env var is missing.
        """
    )

    parser.add_argument('--version', required=True, help='Version number (e.g., 0.1.0)')
    parser.add_argument('--changelog', required=True, help='Changelog description for this version')
    parser.add_argument('--no-github', action='store_true', help='Skip GitHub release and upload')

    args = parser.parse_args()

    if not validate_version_format(args.version):
        print(f"❌ Invalid version format: '{args.version}'. Must be X.Y.Z (e.g., 0.1.0)")
        sys.exit(1)

    # Check for required main script and version file before starting builder
    builder_check = AvaBuilder()  # Temporary instance for path checks
    if not (builder_check.project_root / builder_check.main_script_path).exists():
        print(
            f"❌ Main script '{builder_check.main_script_path}' not found in project root: {builder_check.project_root}")
        print("   Make sure you're running this script from your project root directory.")
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
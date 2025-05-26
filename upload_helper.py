#!/usr/bin/env python3
"""
Simple Upload Helper for AvA Desktop App
========================================

This script makes manual uploading SUPER easy if GITHUB_TOKEN is not set:
1. Opening the GitHub releases page for your repo.
2. Showing you exactly what to copy/paste.
3. Opening the local 'releases' folder where the EXE is.

Usage: python upload_helper.py <version> "<changelog_message>"
Example: python upload_helper.py 0.1.0 "Initial release with core features"
"""

import sys
import webbrowser
import subprocess
from pathlib import Path


def open_upload_helper(version: str, changelog: str):
    """Open everything you need for manual upload"""

    app_name_base = "AvA_DevTool"  # Should match app_name in build_and_deploy.py

    # Determine expected executable name (might need platform-specific adjustments if not .exe)
    # For simplicity, this helper assumes an .exe or a common name pattern.
    # If you build .app.zip for macOS, adjust exe_file_name accordingly.
    exe_file_name = f"{app_name_base}_v{version}.exe"
    # if sys.platform == "darwin":
    #     exe_file_name = f"{app_name_base}_v{version}_macOS.zip" # If you zip .app bundles
    # elif sys.platform.startswith("linux"):
    #     exe_file_name = f"{app_name_base}_v{version}_linux"

    releases_dir = Path("releases")
    exe_file_path = releases_dir / exe_file_name

    # IMPORTANT: Update this to your AvA project's GitHub repository!
    github_user_repo = "carpsesdema/AvA_DevTool"

    print(f"🤖 AvA Dev Tool - Manual Upload Helper for Version {version}")
    print("=" * 70)

    if not exe_file_path.exists():
        print(f"❌ Executable not found: {exe_file_path}")
        print(f"   Ensure you've run the build script first for version {version}:")
        print(f"   python build_and_deploy.py --version {version} --changelog \"{changelog}\" --no-github")
        return

    print(f"✅ Executable ready: {exe_file_path}")
    print(f"📏 Size: {exe_file_path.stat().st_size / (1024 * 1024):.1f} MB")
    print()

    github_releases_new_url = f"https://github.com/{github_user_repo}/releases/new"
    print(f"🌐 Opening GitHub releases page: {github_releases_new_url}")
    webbrowser.open(github_releases_new_url)

    print(f"📁 Opening local releases folder: {releases_dir.resolve()}")
    try:
        if sys.platform == "win32":
            subprocess.run(["explorer", str(releases_dir.resolve())], check=False)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(releases_dir.resolve())], check=False)
        else:  # Linux and other POSIX
            subprocess.run(["xdg-open", str(releases_dir.resolve())], check=False)
    except FileNotFoundError:
        print(f"   Could not automatically open the folder. Please navigate to it manually: {releases_dir.resolve()}")
    except Exception as e:
        print(f"   Error opening folder: {e}")

    print()
    print("📋 COPY THIS INFO TO THE GITHUB RELEASE FORM:")
    print("-" * 40)
    print(f"Tag version:         v{version}")
    print(f"Release title:       {app_name_base} v{version}")
    print(f"Describe this release (changelog):\n{changelog}")
    print("-" * 40)
    print(
        f"File to upload:      Drag '{exe_file_path.name}' from the '{releases_dir.name}' folder into the 'Attach binaries' box on GitHub.")
    print()
    print("🎯 STEPS ON GITHUB:")
    print("1. Fill in the 'Tag version', 'Release title', and 'Describe this release' fields using the info above.")
    print(f"2. Drag & drop the '{exe_file_path.name}' file from the opened folder to the asset upload box.")
    print("3. (Optional) Check 'This is a pre-release' if applicable.")
    print("4. Click 'Publish release'.")
    print("5. Done! Your users can now download this version. 🎉")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {Path(__file__).name} <version> \"<changelog_message>\"")
        print(f"Example: python {Path(__file__).name} 0.1.0 \"Initial release with core features\"")
        sys.exit(1)

    script_version = sys.argv[1]
    script_changelog = sys.argv[2]

    # Basic version validation
    import re

    if not re.fullmatch(r'^\d+\.\d+\.\d+$', script_version):
        print(f"❌ Invalid version format: '{script_version}'. Must be X.Y.Z (e.g., 0.1.0)")
        sys.exit(1)

    open_upload_helper(script_version, script_changelog)
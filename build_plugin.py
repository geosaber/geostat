#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Plugin Script
=====================
Packages the GeoStats plugin into a distributable .zip file
for installation in QGIS Plugin Manager.
"""

import os
import zipfile
import sys

# Directories and files to EXCLUDE from the package
EXCLUDES = {
    "__pycache__",
    ".git",
    ".github",
    ".vscode",
    ".idea",
    "tests",
    "build_plugin.py",
    "PROJECT_MAP.md",
    "RULES.md",
    ".gitignore",
    ".env",
    "venv",
    ".venv",
    ".pytest_cache",
    "node_modules",
}

# File name patterns to exclude
EXCLUDE_FILES = {
    "geostats_plugin.zip",
}

# File extensions to exclude
EXCLUDE_EXTENSIONS = {".pyc", ".pyo", ".bak", ".tmp"}


def build_plugin(plugin_dir: str, output_name: str = "geostats_plugin"):
    """Package the plugin directory into a .zip file.

    Args:
        plugin_dir: Path to the plugin root directory.
        output_name: Name of the output zip (without extension).
    """
    output_path = os.path.join(plugin_dir, f"{output_name}.zip")
    plugin_name = "geostats"  # Name inside the zip

    print(f"Building plugin package: {output_path}")
    print(f"Source directory: {plugin_dir}")

    file_count = 0
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(plugin_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDES]

            for filename in files:
                if filename in EXCLUDES or filename in EXCLUDE_FILES:
                    continue
                _, ext = os.path.splitext(filename)
                if ext in EXCLUDE_EXTENSIONS:
                    continue

                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, plugin_dir)
                arcname = os.path.join(plugin_name, rel_path)

                zf.write(filepath, arcname)
                file_count += 1
                print(f"  + {arcname}")

    print(f"\n[OK] Package created: {output_path}")
    print(f"  Total files: {file_count}")
    print(f"\nInstall in QGIS: Plugins > Manage Plugins > Install from ZIP")


if __name__ == "__main__":
    plugin_dir = os.path.dirname(os.path.abspath(__file__))
    build_plugin(plugin_dir)

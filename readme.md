# CodeWeaver 💎

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform](https://img.shields.io/badge/platform-windows%20%7C%20macos%20%7C%20linux-lightgrey.svg)](#)

**A user-friendly desktop GUI to intelligently package a software codebase into a single, clean Markdown file. This output is optimized for Large Language Models (LLMs) like Claude 3, GPT-4, and other AI tools for advanced code analysis, review, and documentation.**

This tool provides a powerful, visual alternative to command-line context-stuffing scripts, giving you more control and a much smoother workflow for your prompt engineering.

---

## 🌟 Features

*   **🖥️ Intuitive GUI:** No more command-line flags. Manage everything from a clean, responsive user interface.
*   **📂 Project Directory Mode:** Set a main "Projects" folder, and the tool automatically lists all your projects for quick selection.
*   **🌳 Project Structure Tree:** The generated Markdown file begins with a tree-like overview of your entire codebase, giving the AI immediate context of the project's layout.
*   **⚙️ Advanced, Editable Ignore Patterns:** Comes with a comprehensive default list of files and folders to ignore (`node_modules`, `venv`, `.git`, etc.). This list is fully editable in the GUI.
*   **💾 Ignore Profiles:** Save and load different sets of ignore patterns as profiles (e.g., "Python/Django", "Node.js/React"). Comes with several useful presets.
*   **🔬 Live Preview Mode:** See exactly which files will be included in the digest *before* creating the file. Perfect for fine-tuning your ignore patterns.
*   **📊 Output Statistics:** Instantly see the final file size and an estimated token count to ensure your codebase fits within your LLM's context window.
*   **⚡ Efficient & Fast:** Skips entire directories like `venv` or `node_modules` without checking every file inside, making it incredibly fast even on large projects.
*   **🧵 Stable Background Processing:** The entire digest process runs in a background thread, keeping the UI responsive and preventing freezes.
*   **👀 High-DPI Support:** The interface scales correctly and looks sharp on 4K monitors.
*   **🚀 Convenience Actions:** After a digest is created, use the "Show in Explorer" or "Open File" buttons for immediate access to the output.
*   **🔄 Automatic Settings:** Your project directory, ignore list, and custom profiles are automatically saved and loaded between sessions.

## 📸 Screenshot

*(Note: You should replace `screenshot.png` with an actual screenshot of the application.)*

![alt text](screenshot.png)

---

## 🚀 Getting Started

### 1. Prerequisites
*   **Python 3.x** installed on your system.
*   That's it! The script is self-contained and requires **no external libraries** to be installed.

### 2. Run the Application
1.  Save the code as `codeweaver.py`.
2.  Run it from your terminal:
    ```bash
    python codeweaver.py
    ```

### 3. Initial Setup
The first time you run the app, you need to tell it where your projects are located.
1.  Click the **"Set Project Directory"** button.
2.  Choose the main folder that contains all of your code projects (e.g., `C:/Users/You/Documents/Code` or `~/dev`).
3.  The list on the left will now populate with all the subfolders from that directory.

## 📋 How to Use

1.  **Select a project** from the list on the left.
2.  **Adjust the ignore patterns** in the "Settings" panel if needed.
3.  Click the **"Digest Selected Project"** button.
4.  A progress bar will show the status while the log window details which files are being included.
5.  Once complete, a `codebase.md` file will be created inside that project's folder.
6.  Use the **"Show in Explorer"** button to instantly find the file or **"Open File"** to view it.
7.  You can now upload this `codebase.md` file to Claude, a custom GPT's knowledge base, or any other AI tool.

## 🛠️ Advanced Usage

### Ignore Patterns & Profiles
The "Ignore Patterns" text box is the heart of the tool. Add or remove patterns to control what gets included. Each pattern should be on a new line.

*   Patterns ending with a `/` (e.g., `dist/`) will match directories.
*   Wildcards (`*`) are supported (e.g., `*.log`).

Use the **"Profiles"** menu to quickly load presets for different types of projects. To create your own profile, arrange the ignore list as you like, then go to `Profiles -> Save Current as Profile...`.

### Preview Mode
Before running a full digest on a large project, click **"Preview Included Files..."**. This will run the filtering logic and show you a complete list of files that will be included, allowing you to catch any mistakes in your ignore patterns.

## ⚙️ Configuration

The application automatically saves your settings in a JSON file located in your user's home directory:

**File:** `~/.codeweaver_config.json`

This file stores your main project directory path, your custom ignore profiles, and the last-used list of ignore patterns. You can safely delete this file to reset the application to its default state.

## 📄 License

This project is licensed under the Creative Commons BY-NC-SA 4.0 license.
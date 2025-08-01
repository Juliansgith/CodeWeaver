# CodeWeaver 💎

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/codeweaver.svg)](https://badge.fury.io/py/codeweaver)

**CodeWeaver is an AI-native development tool designed to intelligently analyze, select, and package software codebases into context-optimized formats for Large Language Models (LLMs) like Claude 3.5, GPT-4o, and Gemini.**

It bridges the gap between your complex codebase and the context window of an AI, ensuring the model receives the most relevant information for tasks like code analysis, debugging, and documentation.

---

## 🌟 Key Features

*   **🤖 AI-Powered Optimization Engine:** Translates natural language goals (e.g., "debug authentication") into an optimal set of files using semantic search, dependency analysis, and importance scoring.
*   **🗣️ Conversation-Aware Context:** Remembers your conversation history to dynamically adapt the file selection, providing context that evolves with your task.
*   **🔌 MCP Server:** Implements the Model Context Protocol (MCP) to act as an intelligent backend for AI assistants, serving real-time, query-specific context on demand.
*   **🧠 Semantic Search:** Uses Google Gemini embeddings to understand the meaning of your code, finding relevant files even if they don't share keywords with your query.
*   **🔗 Advanced Analysis:** Builds a deep understanding of your project through dependency graph analysis, file importance scoring, and cross-file relationship detection.
*   **🖥️ Intuitive GUI & Powerful CLI:** Manage your workflow through a user-friendly Tkinter-based desktop application or automate it with a comprehensive command-line interface.
*   **📦 Flexible Exports:** Packages the selected context into multiple formats, including Markdown, JSON, HTML, and chunked exports for massive codebases.
*   **⚙️ Smart Templates:** Comes with pre-built templates for common project types (React, Python Backend, etc.) and allows you to create your own via a web-based visual editor.

---

## 🚀 Getting Started

### 1. Installation
Install CodeWeaver and its dependencies from PyPI:
```bash
# For the core CLI and GUI application
pip install codeweaver

# To include the web-based template editor
pip install "codeweaver[web]"

# For all features including advanced exports and analytics
pip install "codeweaver[full]"
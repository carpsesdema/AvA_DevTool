# AvA DevTool: Your AI-Powered Desktop Assistant (PySide6 Rebuild)

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/framework-PySide6-cyan.svg)](https://www.qt.io/qt-for-python)
[![Status](https://img.shields.io/badge/status-Phase%201%20Active-brightgreen.svg)](https://github.com/carpsesdema/AvA_DevTool)
<!-- Add more badges as you see fit: license, build status, etc. -->

**Welcome to AvA DevTool!** 👋

AvA is your intelligent AI desktop partner, designed to assist with a variety of tasks. This project is a rebuild using **PySide6**, focusing on creating a modular, robust, and extensible application. Think of AvA as your evolving sidekick, ready to integrate with powerful AI models and streamline your workflows.

## 🚀 Vision

The goal is to build a versatile assistant that can:

*   Engage in intelligent conversations using various LLMs.
*   Assist with development tasks, including code generation and RAG-based context awareness.
*   Manage projects and chat sessions effectively.
*   Provide a clean, intuitive, and customizable user experience.
*   And much more as the project grows!

## 🌱 Current Status: Phase 1 - Core Chat & UI Foundation

We are currently in **Phase 1**, which lays the groundwork for AvA's capabilities.

**Key Features in Phase 1:**

*   **Modern UI Shell:** Built with PySide6, featuring a `MainWindow` with a `LeftPanel` for controls and a central chat display area.
*   **LLM Integration:**
    *   Support for multiple Large Language Models (LLMs) via a flexible `BackendCoordinator`.
    *   Default integration with **Gemini** for chat. (Support for Ollama & GPT adapters included).
    *   Basic streaming chat functionality for interactive conversations.
*   **Core Chat Functionality:**
    *   "New Chat" session creation.
    *   LLM selection from the `LeftPanel`.
    *   Configuration of AI persona/system prompt.
*   **Developer Tools:**
    *   `LLMCommunicationLog` dialog to inspect AI interactions.
    *   Rudimentary project and session management (persisted locally).
*   **Configuration:** API keys are loaded from a `.env` file for security.

**What's Not Yet Implemented (but on the horizon!):**

*   Advanced project context management or tabbed chat interfaces.
*   Full-fledged Retrieval Augmented Generation (RAG) system.
*   Agentic multi-file workflows (e.g., advanced code bootstrapping, automated file modifications).
*   Sophisticated UI elements like temperature sliders, token counters, etc.
*   Comprehensive session saving/loading beyond the basic current state.

## 🛠️ Getting Started (Phase 1)

Follow these steps to get AvA up and running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/carpsesdema/AvA_DevTool.git
    cd AvA_DevTool
    ```

2.  **Create a Virtual Environment:**
    (Recommended to keep dependencies isolated)
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   On Windows: `venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up API Keys:**
    Create a `.env` file in the project root directory (`AvA_DevTool/`). Add your API keys like this:
    ```env
    # Required for Gemini (default chat LLM)
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

    # Optional: For OpenAI's GPT models
    # OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"

    # Note: Ollama runs locally and typically doesn't require an API key in the .env
    ```
    Replace `"YOUR_GEMINI_API_KEY_HERE"` with your actual Gemini API key.

5.  **Run the Application:**
    ```bash
    python main.py
    ```

AvA should now launch! 🎉

## ✨ Future Phases

AvA is an evolving project. Future phases aim to introduce:

*   **Enhanced RAG Capabilities:** Deeper integration with local files and knowledge bases.
*   **Agentic Workflows:** For more complex task automation and code generation.
*   **Improved UI/UX:** Tabbed chats, advanced configuration options, and a more polished interface.
*   **Plugin System:** To extend AvA's functionality.

Stay tuned for updates!

## 🤝 Contributing (Placeholder)

Details on how to contribute to AvA will be added soon. If you're interested in helping out, please feel free to open an issue or reach out!

---

Enjoy using AvA! We're excited to see how it grows and what it can help you achieve.
If you encounter any bugs or have feature suggestions, please open an issue on GitHub.
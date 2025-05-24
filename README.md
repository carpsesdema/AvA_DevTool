=== README.md ===
# AvA: Advanced Versatile Assistant (PySide6 Rebuild)

Your intelligent AI desktop partner, being rebuilt with PySide6 for a more modular and robust experience!

## Current Phase: Phase 1 - Basic UI Shell & Single LLM Chat

**Focus:**
*   PySide6 `MainWindow` with a `LeftPanel` and a single chat view.
*   `LeftPanel` with LLM selection (Gemini default), "New Chat", "Configure Persona".
*   Basic chat input/display.
*   `BackendCoordinator` with `GeminiAdapter` for streaming chat.
*   `LLMCommunicationLog` dialog.
*   API keys loaded from `.env` via `config.py`.

**Not Implemented in Phase 1:**
*   Project contexts or tabbed chat.
*   RAG system.
*   Multi-file workflows (Bootstrapping, Modification).
*   Session saving/loading (beyond basic last state for persona/model).
*   Advanced UI elements like temperature slider.

## To Run (Phase 1)

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Create a `.env` file** in the project root (`ava_pyside_project/`) with your API key:
    ```env
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```
4.  **Run the application:**
    ```bash
    python main.py
    ```
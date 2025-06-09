# LLM Web Automation Project

## Overview

Advanced web automation system using Playwright with anti-detection capabilities, CAPTCHA solving, computer vision integration, and AI-powered memory for learning automation patterns. Designed for robust and adaptive web scraping and automation tasks.

## Architecture Overview

```text
connector/ (Connection Layer)
├── TaskRunner ← Core orchestration
├── CLI ← Terminal interface
├── API ← REST endpoints
└── UI ← Web interface
↓
web_automation/ (Automation Engine)
├── Browser Agents
├── Memory System (Mem0 + Qdrant + Ollama)
├── Visual System
└── Plugin Architecture
```

## Features

### Core Capabilities

- **Browser Automation**:
  - Headless and non-headless modes via Playwright.
  - Multi-browser support (Chromium, Firefox, WebKit).
  - Human-like interaction patterns.

- **Stealth Mode**:
  - Evade bot detection systems using `playwright-stealth`.
  - Randomized user-agent rotation.
  - Behavioral fingerprint masking.

- **CAPTCHA Handling**:
  - Image-based CAPTCHA solving (placeholder, can integrate with local LLMs).
  - reCAPTCHA v2/v3 support (planned).
  - Audio CAPTCHA fallback (planned).

- **Data Processing**:
  - OCR text extraction.
  - Image analysis with OpenCV.
  - PDF content parsing.

- **Memory Integration (Mem0 AI)**:
  - Utilizes **Mem0 AI** (`mem0ai` library) to store and retrieve successful (and failed) automation patterns (e.g., CSS selectors for specific UI elements).
  - Employs the `PersistentMemoryBrowserAgent` which enhances the base agent with memory capabilities.
  - **Local First**: Configured to run with a local Ollama LLM for processing and a local Qdrant vector store (in-memory or on-disk) for storing embeddings.
  - **Semantic Search**: Uses embeddings and similarity to recall successful patterns when encountering similar UI elements or tasks, improving automation over time.
  - **Robust Memory Handling**: Supports multiple memory formats (dict, list, string) with comprehensive error handling.
  - **Visual Pattern Storage**: Enhanced support for storing and recalling visual patterns for UI elements.
  - **Detailed Documentation**: See [MEMORY_INTEGRATION.md](MEMORY_INTEGRATION.md) for comprehensive documentation.

- **Visual Memory System & Fallback**:
  - Vision-based pattern storage and recall using screenshots and LLM-powered analysis.
  - Robust visual fallback: if a selector fails, the system matches the current UI visually against stored patterns, extracting coordinates for recovery clicks.
  - Type-safe, fully async fallback logic with comprehensive error handling and metadata logging.
  - All visual pattern storage and fallback now use consistent data types, eliminating runtime and type errors.
  - Orphaned and duplicate code removed for maintainability and reliability.
  - Detailed logging and fallback metadata for diagnostics and debugging.

## Web Automation Framework

This project implements a sophisticated web automation framework using Playwright, designed for modularity, scalability, and resilience. The architecture has been refactored to incorporate three key design patterns:

### 1. Plugin System

- **Purpose**: Decouples capabilities from the core agent, allowing for easy addition of new features.
- **Implementation**: Plugins are defined via the `AgentPlugin` abstract base class, with lifecycle methods like `initialize(agent)` and `get_name()`.
- **Examples**: `CaptchaPlugin`, `VisualMemoryPlugin`, and `MemoryPlugin` encapsulate specific functionalities.
- **Usage**: Plugins are passed to the `BrowserAgentFactory` during agent creation and are accessible via `agent.plugins`.

### 2. Command Pattern (Executors)

- **Purpose**: Decouples instruction execution from the core agent logic, enhancing maintainability.
- **Implementation**: Each action type (e.g., `CLICK`, `TYPE`) has a corresponding executor class (e.g., `ClickExecutor`, `TypeExecutor`) inheriting from `BaseExecutor`.
- **Usage**: The agent maintains a dictionary of executors, dynamically dispatching instructions to the appropriate executor.
- **Benefit**: Adding new action types requires only creating a new executor class without modifying the core agent.

### 3. State Machine

- **Purpose**: Enhances resilience by enabling the agent to autonomously handle unexpected states.
- **Implementation**: Defined states include `IDLE`, `EXECUTING`, `CAPTCHA_REQUIRED`, etc., with state checking and transition handling in `PlaywrightBrowserAgent`.
- **Usage**: Before and after instruction execution, the agent checks the page state and transitions to recovery states if needed (e.g., solving CAPTCHAs or dismissing modals).
- **Benefit**: Increases automation robustness by managing interruptions without user intervention.

## Quick Start (5 minutes)

1. **Install & Setup:**

   ```bash
    # Ensure Python 3.8+ is installed
    # Create and activate a virtual environment
    # python -m venv .venv
    # source .venv/bin/activate  (Linux/macOS)
    # .venv\Scripts\activate (Windows)

    pip install -r web_automation/requirements.txt
    # Install dependencies for CLI, API, and UI (if not in the above file)
    pip install typer rich fastapi uvicorn[standard] streamlit pandas
    
    # Install and pull Ollama model (if not already done)
    ollama pull qwen2.5vl:7b
   ```


2. **Try the CLI:**

   ```bash
    python -m connector.cli_interface list-profiles
    # To run a profile, ensure 'example_profile.json' exists in 'connector_cli_logs/profiles/'
    # (or the path configured/created by the CLI). You might need to create one first.
    # python -m connector.cli_interface run-profile example_profile
   ```

3. **Launch the API & UI:**

   *Terminal 1: Start API Server*

   ```bash
    python -m connector.api_interface
   ```

   *Terminal 2: Start Streamlit UI*

   ```bash
    streamlit run ui/app.py
   ```

   Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).

## Connector & Interfaces

The project includes a robust connector layer (`connector/`) to manage and expose the web automation capabilities through various interfaces. This layer is built around the `TaskRunner` class, which handles asynchronous task execution, management, and lifecycle.

### Core: `TaskRunner` (`connector/task_interface.py`)

- Manages asynchronous execution of automation tasks defined by `TaskConfig` (see `connector/models.py`).
- Supports running tasks from predefined JSON profiles or ad-hoc instructions.
- Tracks task status, results, and errors.
- Handles graceful shutdown and resource management.
- Uses `BrowserAgentFactory` to create appropriate browser agents (including memory-enhanced and visual-fallback agents) based on task configuration.

### 1. Command-Line Interface (CLI)

- **Location**: `connector/cli_interface.py`
- **Purpose**: Provides a terminal-based interface for interacting with the `TaskRunner`.
- **Powered by**: `Typer` and `Rich` for a modern CLI experience.

**Key Commands** (run from the project root):

- `python -m connector.cli_interface list-profiles`: Lists available task profiles.
- `python -m connector.cli_interface run-profile <PROFILE_NAME> [--memory-override JSON_STRING] [--visual-override JSON_STRING] [--browser-override JSON_STRING]`: Runs a task from a profile with optional JSON configuration overrides.
- `python -m connector.cli_interface run-instructions <JSON_INSTRUCTIONS_STRING> [--name TASK_NAME] [--description TASK_DESCRIPTION] ...`: Runs a task from ad-hoc JSON instructions.
- `python -m connector.cli_interface status <TASK_ID>`: Checks the status of a task.
- `python -m connector.cli_interface result <TASK_ID>`: Retrieves the result of a completed/failed task.
- `python -m connector.cli_interface cancel <TASK_ID>`: Requests cancellation of a task.
- `python -m connector.cli_interface active-tasks`: Lists all active tasks.

**Setup**:

- Ensure the Python virtual environment is activated.
- The CLI creates a `connector_cli_logs` directory in the project root for its logs and profiles.

### 2. FastAPI REST API

- **Location**: `connector/api_interface.py`
- **Purpose**: Exposes `TaskRunner` functionalities via a RESTful API.
- **Powered by**: `FastAPI`.

**Running the API** (from the project root):

  ```bash
  python -m connector.api_interface
  ```

  The API will typically be available at `http://localhost:8000`.
  Interactive API documentation (Swagger UI) is available at `http://localhost:8000/docs`.

**Key Endpoints**:

- `GET /health`: API health check.
- `GET /profiles`: List available task profiles.
- `POST /tasks/run-profile/{profile_name}`: Submit task from profile (supports JSON body for overrides).
- `POST /tasks/run-instructions`: Submit task from ad-hoc instructions (JSON body).
- `GET /tasks/{task_id}/status`: Get task status.
- `GET /tasks/{task_id}/result`: Get task result.
- `POST /tasks/{task_id}/cancel`: Cancel task.
- `GET /tasks/active`: List active tasks.

**Example API Calls:**

```bash
# List available task profiles
curl http://localhost:8000/profiles

# Run a task from a profile (e.g., 'example_profile')
# Ensure 'example_profile.json' exists where the API expects it (e.g., 'api_connector_logs/profiles/')
curl -X POST http://localhost:8000/tasks/run-profile/example_profile \
  -H "Content-Type: application/json" \
  -d '{
        "browser_config_override": {"headless": false},
        "memory_config_override": {"enabled": true}
      }'

# Run a task with ad-hoc instructions
curl -X POST http://localhost:8000/tasks/run-instructions \
  -H "Content-Type: application/json" \
  -d '{
        "task_config": {
          "name": "AdHoc Google Search via API",
          "description": "Performs a search on Google and captures a screenshot.",
          "profile_config": {
            "memory_config": { "enabled": false },
            "visual_config": { "enabled": false },
            "browser_config": { "browser_type": "chromium", "headless": true }
          }
        },
        "instructions": [
          { "action": "NAVIGATE", "url": "https://www.google.com" },
          { "action": "TYPE", "selector": "textarea[name=q]", "text": "Automated Web LLM with FastAPI" },
          { "action": "PRESS", "selector": "textarea[name=q]", "key": "Enter" },
          { "action": "WAIT_FOR_TIMEOUT", "timeout": 3000 },
          { "action": "SCREENSHOT", "path": "api_google_search_results.png" }
        ]
      }'

# Check status of a task (replace YOUR_TASK_ID with an actual ID)
# curl http://localhost:8000/tasks/YOUR_TASK_ID/status

# Get result of a task (replace YOUR_TASK_ID)
# curl http://localhost:8000/tasks/YOUR_TASK_ID/result
```

**Setup**:

- The API creates an `api_connector_logs` directory in the project root for its logs and profiles.

### 3. Streamlit Web UI

- **Location**: `ui/app.py`
- **Purpose**: Provides a graphical user interface for interacting with the automation system via the FastAPI backend.
- **Powered by**: `Streamlit`.

**Running the UI** (from the project root):

  1. Ensure the FastAPI server (`python -m connector.api_interface`) is running.
  2. In a new terminal, activate the virtual environment and run:

     ```bash
     streamlit run ui/app.py
     ```

**Features**:

- View available task profiles.
- Submit tasks from profiles with optional configuration overrides.
- Submit tasks using ad-hoc JSON instructions with optional configurations.
- View active tasks.
- Query status and results of specific tasks.
- Request task cancellation.

## Advanced Setup

### Dependencies

- **Python**: Recommended Python version 3.8+
- **Core Python Packages:**
  - `playwright`
  - `playwright-stealth`
  - `pydantic`
  - `python-dotenv`
  - `pillow`
  - `opencv-python-headless`
  - `numpy`
  - `pytest`, `pytest-asyncio`, `pytest-mock`
  - `python-json-logger`
  - `requests`
  - `python-multipart`
  - `pyyaml`
  - `mem0ai` (AI memory integration)
  - `qdrant-client` (vector store)
  - `sentence-transformers` (embeddings)
  - `typer` (for CLI)
  - `rich` (for CLI)
  - `fastapi` (for REST API)
  - `uvicorn[standard]` (ASGI server for FastAPI)
  - `streamlit` (for Web UI)
  - `pandas` (used in Streamlit UI)

- **External Dependencies:**
  - **Ollama** (for local LLM): [ollama.com](https://ollama.com/)
    - Ensure a supported model (e.g., `qwen2.5vl:7b`) is pulled and the Ollama server is running locally.
  - **System Dependencies (Windows Example):**
    ```bash
    choco install -y vcredist2015
    ```

### Environment Configuration

1. **Ollama Setup:**
   - Install Ollama from [ollama.com](https://ollama.com/).
   - Pull a multimodal or standard LLM for Mem0 and vision tasks:
     ```bash
     ollama pull qwen2.5vl:7b
     ```
   - Ensure Ollama server is running (default: `http://localhost:11434`).

2. **Python Virtual Environment & Dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r web_automation/requirements.txt
   ```

3. **.env File (Optional):**
   - If any environment variables are needed, create a `.env` file in the project root and add them as `KEY=VALUE` pairs.
   - For local Ollama, API keys are not required. If using OpenAI or other providers, add your API key as `OPENAI_API_KEY=...`.
   - **Security:** `.env` is in `.gitignore` by default and should never be committed.

### Testing

- **Run the test suite:**
  ```bash
  pytest web_automation/tests/
  ```
- **Requirements:**
  - Ollama must be running with the required model pulled.
  - Qdrant is used in in-memory mode for tests (no external setup required).
  - All dependencies from `requirements.txt` must be installed.
- **Integration tests** validate real memory and vision workflows using Playwright, Mem0, Qdrant, and Ollama.

### Extending the Framework
- Add new plugins by subclassing `AgentPlugin`.
- Add new action types by creating new executor classes.
- Define new states in `AgentState` enum for additional resilience scenarios.

## Getting Started

1. **Environment Setup**:
    - Ensure Python (3.8+ recommended) and external dependencies like Ollama are installed (see "Advanced Setup" for details).
    - Create a Python virtual environment and activate it.
    - Install Python dependencies. It's recommended to consolidate all project dependencies into a root `requirements.txt` file. For now, install core web automation dependencies and any others required for the connector/UI:

        ```bash
        pip install -r web_automation/requirements.txt 
        # Ensure FastAPI, Uvicorn, Typer, Rich, Streamlit, Pandas are also installed if not in the above file.
        # Example: pip install fastapi uvicorn[standard] typer rich streamlit pandas
        ```

    - Configure your `.env` file in the project root if necessary (e.g., for API keys if not using local Ollama).

2. **Running the System**:
    - **Programmatic Use**: See the "Usage" section for examples of how to use the `BrowserAgentFactory` and agents in your Python scripts.
    - **CLI**: Activate your virtual environment and use `python -m connector.cli_interface --help` to explore commands.
    - **API**: Activate your virtual environment, then run `python -m connector.api_interface`. Access the API at `http://localhost:8000` and docs at `http://localhost:8000/docs`.
    - **Streamlit UI**: First, ensure the API backend is running (`python -m connector.api_interface`). Then, in a new terminal (with venv activated), run `streamlit run ui/app.py`.

3. **Testing**:
    - Execute tests with `pytest web_automation/tests/` (ensure Ollama is running for tests that require it).

4. **Extending the Framework**:
    - Follow guidelines in "Web Automation Framework" and "Advanced Setup" for adding plugins, executors, or states.

## Usage

Using the memory-enhanced agent:

```python
from web_automation.core.browser_agent_factory import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig
import asyncio
import os

async def main():
    # Configure Mem0 for local Ollama and in-memory Qdrant
    mem_config = Mem0AdapterConfig(
        llm_provider="ollama",
        llm_model=os.getenv("MEMORY_LLM_MODEL"), 
        qdrant_on_disk=False, # In-memory
        qdrant_embedding_model_dims=384
    )

    agent = create_playwright_agent(
        memory_enabled=True,
        headless=True, 
        memory_config=mem_config
    )

    async with agent:
        await agent.navigate("https://example.com")
        # Example of using smart_selector_click which leverages memory
        await agent.smart_selector_click("login button")

if __name__ == "__main__":
    asyncio.run(main())

## Workflow Examples
(Existing examples can be adapted to use `PersistentMemoryBrowserAgent` or the factory method `create_playwright_agent` with `memory_enabled=True`.)

### Basic Scraping (with potential memory use)
```python
# (Similar to above, but agent could be PersistentMemoryBrowserAgent)
```

## Troubleshooting
### Common Issues

| Issue | Solution |
|-------|----------|
| CAPTCHA not solved | Verify CAPTCHA solver configuration. For vision, ensure Ollama and the specified model are running. |
| Detection triggers | Ensure stealth mode is active. |
| Browser crashes | Check memory settings and system resources. |
| `NameError: name 'pytz' is not defined` | Ensure `import pytz` is present and `pytz` is in `requirements.txt` and installed. |
| Mem0 `add` or `search` issues | Verify `Mem0AdapterConfig` (LLM, embedder, Qdrant settings). Ensure Ollama is running and accessible. Check Qdrant logs if on-disk. Increase `llm_temperature` in `Mem0AdapterConfig` if Mem0 logs 'No new facts retrieved'. |
| Test failures related to Mem0 | Ensure unique Qdrant collection names per test, `qdrant_on_disk=False` for in-memory tests, and correct `agent_id` propagation. |

## Folder Structure

- `web_automation/core`: Main automation logic, browser agent, factory.
- `web_automation/memory`: Mem0 AI integration, memory-enhanced agent (`PersistentMemoryBrowserAgent`), memory manager (`Mem0BrowserAdapter`).
- `web_automation/captcha`: CAPTCHA solving modules.
- `web_automation/config`: Settings, Pydantic configuration models, and profiles.
- `web_automation/utils`: Helper functions.
- `web_automation/profiles`: JSON profiles for automation workflows.
- `web_automation/tests`: Pytest integration and unit tests.

## Contributing

We welcome contributions to enhance this framework. Please refer to the architecture sections above to understand the design patterns in use. For major changes, open an issue to discuss your ideas.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

# LLM Web Automation Project

## Overview

Advanced web automation system using Playwright with anti-detection capabilities, CAPTCHA solving, computer vision integration, and AI-powered memory for learning automation patterns. Designed for robust and adaptive web scraping and automation tasks.

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

## Advanced Setup

### Environment Configuration

1. **Python**: Recommended Python version 3.8+.

2. **System Dependencies (Windows Example)**:

   ```bash
   choco install -y vcredist2015
   ```

3. **Ollama (for local LLM)**:
   - Install Ollama from [ollama.com](https://ollama.com/).
   - Pull a multimodal LLM if you intend to use vision-based CAPTCHA solving or other vision tasks, or a standard LLM for Mem0 fact extraction:

     ```bash
     ollama pull ${CAPTCHA_MODEL} # Example multimodal model
     ```

   - Ensure Ollama server is running.

4. **Virtual Environment & Python Dependencies**:

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment (Windows)
   .venv\Scripts\activate

   # Install dependencies
   pip install -r web_automation/requirements.txt
   ```

   Key dependencies in `requirements.txt` include:
   - `playwright`
   - `playwright-stealth`
   - `mem0ai~=1.1`
   - `qdrant-client`
   - `sentence-transformers` (for embeddings)
   - `ollama` (Python client for Ollama)
   - `pytz`
   - `opencv-python`
   - `pydantic`

## Getting Started

1. **Setup**: Install dependencies with `pip install -r requirements.txt`.
2. **Configuration**: Adjust settings in `.env` for models, browser configurations, etc.
3. **Running Tests**: Execute tests with `pytest web_automation/tests/` to validate plugin, executor, and state machine functionality.
4. **Extending the Framework**:
   - Add new plugins by subclassing `AgentPlugin`.
   - Add new action types by creating new executor classes.
   - Define new states in `AgentState` enum for additional resilience scenarios.

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

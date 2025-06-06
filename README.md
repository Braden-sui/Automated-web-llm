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
  - Image-based CAPTCHA solving (placeholder, can integrate with local LLMs like `qwen2.5vl:7b`).
  - reCAPTCHA v2/v3 support (planned).
  - Audio CAPTCHA fallback (planned).

- **Data Processing**:
  - OCR text extraction.
  - Image analysis with OpenCV.
  - PDF content parsing.

- **Memory Integration (Mem0 AI)**:
  - Utilizes **Mem0 AI** (`mem0ai` library) to store and retrieve successful (and failed) automation patterns (e.g., CSS selectors for specific UI elements).
  - Employs the `PersistentMemoryBrowserAgent` which enhances the base agent with memory capabilities.
  - **Local First**: Configured to run with a local Ollama LLM (e.g., `qwen2.5vl:7b`) for processing and a local Qdrant vector store (in-memory or on-disk) for storing embeddings.
  - This allows the agent to learn from past interactions and improve its selector strategy over time without relying on external cloud services for core memory functions.

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
     ollama pull qwen2.5vl:7b # Example multimodal model
     ollama pull llama3       # Example general LLM
     ```

   - Ensure Ollama server is running.

4. **Virtual Environment & Python Dependencies**:

   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate environment (Windows)
   .venv\Scripts\activate
   # (macOS/Linux)
   # source .venv/bin/activate

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

### Configuration Files

1. **Environment Variables (`.env`)**: Loaded from a `.env` file in the project root. Used for sensitive info and high-level settings.
   Example for CAPTCHA vision model (if using local Ollama for vision):

   ```env
   CAPTCHA_VISION_MODEL_NAME=qwen2.5vl:7b
   CAPTCHA_VISION_OLLAMA_URL=http://localhost:11434
   ```

2. **Main Python Configuration (`web_automation/config/settings.py` and `web_automation/config/config_models.py`)**:
   - `settings.py`: General application settings (e.g., `BROWSER_HEADLESS`).
   - `config_models.py`: Contains Pydantic models for structured configuration, including `Mem0AdapterConfig`.
     - `Mem0AdapterConfig` defines the setup for Mem0 AI, including:
       - LLM provider (e.g., `ollama`), model name (e.g., `qwen2.5vl:7b`), temperature.
       - Embedder provider (`huggingface`), model (`sentence-transformers/all-MiniLM-L6-v2`), and dimensions (384).
       - Vector store provider (`qdrant`), collection name, and whether it runs in-memory or on-disk.

## Usage

Using the memory-enhanced agent:

```python
from web_automation.core.browser_agent_factory import create_playwright_agent
from web_automation.config.config_models import Mem0AdapterConfig
import asyncio

async def main():
    # Configure Mem0 for local Ollama and in-memory Qdrant
    mem_config = Mem0AdapterConfig(
        llm_provider="ollama",
        llm_model="qwen2.5vl:7b", # Or your preferred Ollama model
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
        success = await agent.smart_selector_click(
            target_description="Main link on example.com",
            fallback_selector="a[href='https://www.iana.org/domains/example']"
        )
        print(f"Click successful: {success}")

if __name__ == "__main__":
    asyncio.run(main())
```

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

## Contribution Guidelines

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature-name`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a Pull Request.

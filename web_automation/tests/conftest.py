import os
import pytest
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b")


def is_ollama_server_ready(base_url=OLLAMA_BASE_URL):
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

def is_ollama_model_available(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL):
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=2)
        if resp.status_code != 200:
            return False
        tags = resp.json().get("models", [])
        return any(model in (m.get("name") or "") for m in tags)
    except Exception:
        return False

@pytest.fixture(scope="session", autouse=True)
def ensure_ollama_ready():
    """
    Ensure Ollama server and required model are available before running tests.
    Skips all tests if not available.
    """
    if not is_ollama_server_ready():
        pytest.skip("Ollama server is not running at {}. Skipping tests that require Ollama.".format(OLLAMA_BASE_URL))
    if not is_ollama_model_available():
        pytest.skip(f"Ollama model '{OLLAMA_MODEL}' is not available. Run 'ollama pull {OLLAMA_MODEL}' and try again.")

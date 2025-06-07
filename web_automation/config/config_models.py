from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os

class BrowserConfig(BaseModel):
    DEFAULT_BROWSER_TYPE: str = Field("chromium", description="Default browser type (chromium, firefox, webkit)")
    DEFAULT_HEADLESS_MODE: bool = Field(True, description="Default headless mode for the browser")
    DEFAULT_VIEWPORT_WIDTH: int = Field(1280, description="Default viewport width")
    DEFAULT_VIEWPORT_HEIGHT: int = Field(720, description="Default viewport height")
    USER_AGENTS_FILE: str = Field("web_automation/config/user_agents.json", description="Path to the user agents JSON file")
    EXTRA_BROWSER_ARGS: List[str] = Field([], description="Extra arguments to pass to the browser launch command")
    DISABLE_JAVASCRIPT: bool = Field(False, description="Disable JavaScript execution in the browser")

class ProxyConfig(BaseModel):
    USE_PROXY: bool = Field(False, description="Whether to use a proxy server")
    PROXY_SERVER: Optional[str] = Field(None, description="Proxy server address (e.g., http://localhost:8080)")
    PROXY_USERNAME: Optional[str] = Field(None, description="Username for proxy authentication")
    PROXY_PASSWORD: Optional[str] = Field(None, description="Password for proxy authentication")

class AntiDetectionConfig(BaseModel):
    RANDOMIZE_FINGERPRINT: bool = Field(True, description="Randomize browser fingerprint on each launch")
    CUSTOM_FINGERPRINT_SCRIPT: Optional[str] = Field(None, description="Path to a custom JavaScript file for fingerprint modification")
    MIN_DELAY_BEFORE_ACTION: int = Field(500, description="Minimum delay in ms before performing an action")
    MAX_DELAY_BEFORE_ACTION: int = Field(1500, description="Maximum delay in ms before performing an action")
    MIN_DELAY_AFTER_ACTION: int = Field(200, description="Minimum delay in ms after performing an action")
    MAX_DELAY_AFTER_ACTION: int = Field(800, description="Maximum delay in ms after performing an action")
    MIN_TYPING_DELAY_PER_CHAR: int = Field(50, description="Minimum delay in ms per character typed")
    MAX_TYPING_DELAY_PER_CHAR: int = Field(150, description="Maximum delay in ms per character typed")

class GeneralConfig(BaseModel):
    DEFAULT_TIMEOUT: int = Field(30000, description="Default timeout in ms for Playwright operations")
    DOWNLOADS_DIR: str = Field("downloads", description="Directory to save downloaded files")
    SCREENSHOTS_DIR: str = Field("screenshots", description="Directory to save screenshots")
    LOG_LEVEL: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    PROFILES_DIR: str = Field("web_automation/profiles", description="Directory to store user profiles")

class Mem0AdapterConfig(BaseModel):
    """
    Configuration for Mem0 AI adapter for browser automation memory.
    Includes LLM, embedder, and vector store settings.
    """
    # Qdrant Configuration
    qdrant_path: Optional[str] = Field(None, description="Path to Qdrant DB if on-disk. None for in-memory.")
    qdrant_on_disk: bool = Field(False, description="Whether to store Qdrant DB on disk (True) or in-memory (False).")
    qdrant_collection_name: str = Field("browser_automation_memory", description="Collection name for Qdrant vector store.")
    qdrant_embedding_model_dims: int = Field(384, description="Dimensions of embedding model for Qdrant (e.g., 384 for all-MiniLM-L6-v2).")
    
    # Mem0 Version
    mem0_version: str = Field("v1.1", description="Mem0 AI library version compatibility (v1.0 or v1.1).")
    
    # LLM Configuration
    llm_provider: str = Field("ollama", description="LLM provider for Mem0 (e.g., 'ollama', 'openai').")
    llm_model: str = Field(default=os.getenv("MEMORY_LLM_MODEL", "qwen2.5vl:7b"), description="LLM model name for Mem0 (e.g., 'qwen2.5vl:7b' for Ollama). Ensure the model is available for the chosen provider.")
    llm_temperature: float = Field(0.7, description="Temperature for LLM responses. Lower for factual extraction.")
    api_key: Optional[str] = Field(None, description="API key if required by provider (None for local Ollama).")
    agent_id: Optional[str] = Field(
        None, 
        description="Specific agent ID for Mem0 AI, if applicable."
    )
    qdrant_collection_name: Optional[str] = Field(
        "mem0_default_collection", 
        description="Name of the Qdrant collection to use. Helps isolate test data."
    )
    llm_base_url: Optional[str] = Field(None, description="Base URL for the LLM provider, if not default (e.g., for a self-hosted Ollama instance on a different port/host).")

class VisualSystemConfig(BaseModel):
    """
    Visual system configuration.
    By default, visual memory is disabled and auto-capture is off. Enable explicitly for image analysis or fallback.
    """
    enabled: bool = Field(False, description="Enable or disable visual pattern recognition features.")
    auto_capture: bool = Field(False, description="Automatically capture and store screenshots during automation. Default False.")
    model_name: str = Field(default=os.getenv("VISUAL_SYSTEM_MODEL", "qwen2.5vl:7b"), description="The Ollama vision model name to use (e.g., 'qwen2.5vl:7b', 'llava').")
    ollama_base_url: Optional[str] = Field(None, description="Optional base URL for the Ollama server if not running on default localhost:11434.")

class Settings(BaseModel):
    browser_config: BrowserConfig = Field(default_factory=BrowserConfig)
    proxy_config: ProxyConfig = Field(default_factory=ProxyConfig)
    anti_detection_config: AntiDetectionConfig = Field(default_factory=AntiDetectionConfig)
    general_config: GeneralConfig = Field(default_factory=GeneralConfig)
    mem0ai_config: Mem0AdapterConfig = Field(default_factory=Mem0AdapterConfig)
    visual_system_config: VisualSystemConfig = Field(default_factory=VisualSystemConfig) # Added visual config

# Example of how settings might be loaded (e.g., in a main config file or __init__.py)
# settings = Settings()
# You might load these from a .env file or a YAML/JSON config file using Pydantic's capabilities.

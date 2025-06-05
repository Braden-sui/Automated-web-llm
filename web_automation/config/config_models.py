from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

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
    api_key: Optional[str] = Field(
        None, 
        description="API key for Mem0 AI. If None, mem0ai might use environment variables."
    )
    agent_id: Optional[str] = Field(
        None, 
        description="Specific agent ID for Mem0 AI, if applicable."
    )
    
    # Qdrant specific configuration for Mem0's vector store
    qdrant_path: Optional[str] = Field(
        None, 
        description="Path for Qdrant local storage. Set to None for in-memory (if on_disk is False and path is None)."
    )
    qdrant_on_disk: Optional[bool] = Field(
        False, 
        description="Whether Qdrant should store data on disk. If False and path is None, attempts in-memory."
    )
    qdrant_collection_name: Optional[str] = Field(
        "mem0_default_collection", 
        description="Name of the Qdrant collection to use. Helps isolate test data."
    )
    qdrant_embedding_model_dims: Optional[int] = Field(
        None, 
        description="Dimension for Qdrant collection vectors, e.g., 384 for all-MiniLM-L6-v2. Corresponds to 'embedding_model_dims' in Qdrant config via Mem0."
    )
    mem0_version: str = Field("v1.1", description="Mem0 configuration version.")
    llm_provider: str = Field("openai", description="LLM provider for Mem0 (e.g., 'openai', 'ollama').")
    llm_model: str = Field("gpt-4o-mini", description="LLM model name for Mem0 (e.g., 'gpt-4o-mini' for OpenAI, 'llama3' for Ollama). Ensure the model is available for the chosen provider.")
    llm_temperature: float = Field(0.1, description="LLM temperature for Mem0.")
    llm_base_url: Optional[str] = Field(None, description="Base URL for the LLM provider, if not default (e.g., for a self-hosted Ollama instance on a different port/host).")
    # Future consideration: host/port for remote Qdrant
    # qdrant_host: Optional[str] = Field(None, description="Hostname for a remote Qdrant instance.")
    # qdrant_port: Optional[int] = Field(None, description="Port for a remote Qdrant instance.")

    # Placeholder for other Mem0 specific configurations if needed
    # model_name: Optional[str] = Field(None, description="Model to use for Mem0 AI.")
    # extra_params: Dict[str, Any] = Field({}, description="Any extra parameters for Mem0 initialization.")

class Settings(BaseModel):
    browser_config: BrowserConfig = Field(default_factory=BrowserConfig)
    proxy_config: ProxyConfig = Field(default_factory=ProxyConfig)
    anti_detection_config: AntiDetectionConfig = Field(default_factory=AntiDetectionConfig)
    general_config: GeneralConfig = Field(default_factory=GeneralConfig)
    mem0ai_config: Mem0AdapterConfig = Field(default_factory=Mem0AdapterConfig) # New config

# Example of how settings might be loaded (e.g., in a main config file or __init__.py)
# settings = Settings()
# You might load these from a .env file or a YAML/JSON config file using Pydantic's capabilities.

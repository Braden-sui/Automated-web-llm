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
    api_key: Optional[str] = Field(None, description="API key for Mem0 AI. If None, mem0ai might use environment variables.")
    agent_id: Optional[str] = Field(None, description="Specific agent ID for Mem0 AI, if applicable.")
    # Add other relevant Mem0 configuration parameters here
    # e.g., model_name: Optional[str] = Field(None, description="Model to use for Mem0 AI.")
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

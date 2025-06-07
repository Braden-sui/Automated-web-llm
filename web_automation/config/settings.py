# web_automation/config/settings.py
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from web_automation.config.config_models import Settings as PydanticSettings # Alias to avoid naming conflict

load_dotenv()  # Load environment variables from .env file

class AntiDetectionConfig:
    """Configuration for anti-detection measures"""
    
    # Delays in milliseconds
    MIN_DELAY_BEFORE_ACTION = int(os.getenv("MIN_DELAY_BEFORE_ACTION", "100"))
    MAX_DELAY_BEFORE_ACTION = int(os.getenv("MAX_DELAY_BEFORE_ACTION", "300"))
    
    MIN_DELAY_AFTER_ACTION = int(os.getenv("MIN_DELAY_AFTER_ACTION", "200"))
    MAX_DELAY_AFTER_ACTION = int(os.getenv("MAX_DELAY_AFTER_ACTION", "500"))
    
    MIN_TYPING_DELAY_PER_CHAR = int(os.getenv("MIN_TYPING_DELAY_PER_CHAR", "50"))
    MAX_TYPING_DELAY_PER_CHAR = int(os.getenv("MAX_TYPING_DELAY_PER_CHAR", "150"))
    
    # Mouse movement
    RANDOMIZE_MOUSE_PATHS = os.getenv("RANDOMIZE_MOUSE_PATHS", "True").lower() == "true"
    CLICK_POSITION_VARIANCE = int(os.getenv("CLICK_POSITION_VARIANCE", "5"))  # pixels
    
    # Stealth settings
    USE_STEALTH_PLUGIN = os.getenv("USE_STEALTH_PLUGIN", "True").lower() == "true"
    RANDOMIZE_VIEWPORT = os.getenv("RANDOMIZE_VIEWPORT", "True").lower() == "true"
    RANDOMIZE_USER_AGENT = os.getenv("RANDOMIZE_USER_AGENT", "True").lower() == "true"
    
    # Behavioral patterns
    SIMULATE_HUMAN_PAUSES = os.getenv("SIMULATE_HUMAN_PAUSES", "True").lower() == "true"
    PAUSE_PROBABILITY = float(os.getenv("PAUSE_PROBABILITY", "0.15"))  # 15% chance
    MIN_PAUSE_DURATION = int(os.getenv("MIN_PAUSE_DURATION", "500"))
    MAX_PAUSE_DURATION = int(os.getenv("MAX_PAUSE_DURATION", "2000"))

class ExternalCaptchaServiceConfig:
    """Configuration for CAPTCHA solving services"""
    
    # API Keys
    TWOCAPTCHA_API_KEY = os.getenv("TWOCAPTCHA_API_KEY")
    ANTICAPTCHA_API_KEY = os.getenv("ANTICAPTCHA_API_KEY")
    CAPMONSTER_API_KEY = os.getenv("CAPMONSTER_API_KEY")
    
    # Service settings
    DEFAULT_CAPTCHA_SERVICE = os.getenv("DEFAULT_CAPTCHA_SERVICE", "2captcha")
    CAPTCHA_SOLVE_TIMEOUT = int(os.getenv("CAPTCHA_SOLVE_TIMEOUT", "180"))  # seconds
    CAPTCHA_RETRY_LIMIT = int(os.getenv("CAPTCHA_RETRY_LIMIT", "3"))
    
    # Automatic detection
    AUTO_DETECT_CAPTCHA = os.getenv("AUTO_DETECT_CAPTCHA", "True").lower() == "true"
    CAPTCHA_DETECTION_TIMEOUT = int(os.getenv("CAPTCHA_DETECTION_TIMEOUT", "5000"))  # ms
    
    # Fallback options
    ENABLE_HUMAN_FALLBACK = os.getenv("ENABLE_HUMAN_FALLBACK", "False").lower() == "true"
    HUMAN_FALLBACK_TIMEOUT = int(os.getenv("HUMAN_FALLBACK_TIMEOUT", "300"))  # seconds

class BrowserConfig:
    """Browser-specific configuration"""
    
    DEFAULT_BROWSER_TYPE = os.getenv("DEFAULT_BROWSER_TYPE", "chromium")
    DEFAULT_HEADLESS_MODE = os.getenv("DEFAULT_HEADLESS_MODE", "False").lower() == "true"  # Changed default
    
    # Viewport settings
    DEFAULT_VIEWPORT_WIDTH = int(os.getenv("DEFAULT_VIEWPORT_WIDTH", "1920"))
    DEFAULT_VIEWPORT_HEIGHT = int(os.getenv("DEFAULT_VIEWPORT_HEIGHT", "1080"))
    
    # Browser arguments
    EXTRA_BROWSER_ARGS = os.getenv("EXTRA_BROWSER_ARGS", "").split(",") if os.getenv("EXTRA_BROWSER_ARGS") else []
    
    # User agents (simplified list)
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]

class GeneralConfig:
    """General application configuration"""
    
    # Timeouts
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30000"))  # milliseconds
    PAGE_LOAD_TIMEOUT = int(os.getenv("PAGE_LOAD_TIMEOUT", "60000"))
    ELEMENT_WAIT_TIMEOUT = int(os.getenv("ELEMENT_WAIT_TIMEOUT", "10000"))
    
    # Directories
    SCREENSHOTS_DIR = os.getenv("SCREENSHOTS_DIR", "screenshots")
    DOWNLOADS_DIR = os.getenv("DOWNLOADS_DIR", "downloads")
    LOGS_DIR = os.getenv("LOGS_DIR", "logs")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "True").lower() == "true"
    MAX_LOG_FILE_SIZE = int(os.getenv("MAX_LOG_FILE_SIZE", "10485760"))  # 10MB
    
    # Performance
    MAX_CONCURRENT_BROWSERS = int(os.getenv("MAX_CONCURRENT_BROWSERS", "3"))
    ENABLE_PERFORMANCE_MONITORING = os.getenv("ENABLE_PERFORMANCE_MONITORING", "True").lower() == "true"
    
    # Retry settings
    DEFAULT_RETRY_COUNT = int(os.getenv("DEFAULT_RETRY_COUNT", "3"))
    RETRY_DELAY_BASE = int(os.getenv("RETRY_DELAY_BASE", "1000"))  # ms, exponential backoff

class ProxyConfig:
    """Proxy configuration for additional anonymity"""
    
    USE_PROXY = os.getenv("USE_PROXY", "False").lower() == "true"
    PROXY_SERVER = os.getenv("PROXY_SERVER")  # e.g., "http://proxy.example.com:8080"
    PROXY_USERNAME = os.getenv("PROXY_USERNAME")
    PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")
    
    # Proxy rotation
    ENABLE_PROXY_ROTATION = os.getenv("ENABLE_PROXY_ROTATION", "False").lower() == "true"
    PROXY_LIST_FILE = os.getenv("PROXY_LIST_FILE", "proxies.txt")

class SecurityConfig:
    """Security and privacy settings"""
    
    # Data retention
    AUTO_DELETE_SCREENSHOTS = os.getenv("AUTO_DELETE_SCREENSHOTS", "True").lower() == "true"
    SCREENSHOT_RETENTION_DAYS = int(os.getenv("SCREENSHOT_RETENTION_DAYS", "7"))
    
    # Privacy
    DISABLE_IMAGES = os.getenv("DISABLE_IMAGES", "False").lower() == "true"
    DISABLE_JAVASCRIPT = os.getenv("DISABLE_JAVASCRIPT", "False").lower() == "true"
    BLOCK_ADS = os.getenv("BLOCK_ADS", "True").lower() == "true"
    
    # Rate limiting
    ENABLE_RATE_LIMITING = os.getenv("ENABLE_RATE_LIMITING", "True").lower() == "true"
    REQUESTS_PER_MINUTE = int(os.getenv("REQUESTS_PER_MINUTE", "30"))

# AWM (Agent Workflow Memory) has been removed in favor of Mem0 integration

# Instantiate configs for easy import

class CaptchaConfig(BaseModel):
    """Configuration for local vision-based CAPTCHA solving"""
    ENABLED: bool = os.getenv("CAPTCHA_ENABLED", "True").lower() == "true"
    CAPTCHA_VISION_MODEL_NAME: str = os.getenv("CAPTCHA_MODEL", "qwen2.5-vl:7b")
    CAPTCHA_MAX_ATTEMPTS: int = int(os.getenv("CAPTCHA_MAX_ATTEMPTS", "3"))
    SOLVE_TIMEOUT_SECONDS: int = int(os.getenv("CAPTCHA_SOLVE_TIMEOUT_SECONDS", "30"))
    USE_OCR_FALLBACK: bool = os.getenv("CAPTCHA_USE_OCR_FALLBACK", "True").lower() == "true"
    PREPROCESS_IMAGES: bool = os.getenv("CAPTCHA_PREPROCESS_IMAGES", "True").lower() == "true"
    CAPTCHA_RETRY_DELAY_MS: int = int(os.getenv("CAPTCHA_RETRY_DELAY_MS", "2000"))
    CAPTCHA_POST_SUBMIT_DELAY_MS: int = int(os.getenv("CAPTCHA_POST_SUBMIT_DELAY_MS", "3000"))
    _default_success_texts = "captcha solved,verification successful,challenge complete,thank you"
    CAPTCHA_SUCCESS_TEXTS: List[str] = [
            s.strip() for s in os.getenv("CAPTCHA_SUCCESS_TEXTS", _default_success_texts).split(',') if s.strip()
        ]
    _default_failure_texts = "incorrect captcha,verification failed,try again,error,invalid code,challenge failed"
    CAPTCHA_FAILURE_TEXTS: List[str] = [
            s.strip() for s in os.getenv("CAPTCHA_FAILURE_TEXTS", _default_failure_texts).split(',') if s.strip()
        ]


anti_detection_config = AntiDetectionConfig()
external_captcha_service_config = ExternalCaptchaServiceConfig()
browser_config = BrowserConfig()
general_config = GeneralConfig()
proxy_config = ProxyConfig()
captcha_config = CaptchaConfig() # Local vision CAPTCHA config
security_config = SecurityConfig()

# Instantiate Pydantic-based settings
# This will load .env variables for Pydantic models if not explicitly passed
pydantic_settings = PydanticSettings()
mem0_adapter_config = pydantic_settings.mem0ai_config

# Ensure necessary directories exist
for directory in [general_config.SCREENSHOTS_DIR, general_config.DOWNLOADS_DIR, general_config.LOGS_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Validation functions
def validate_config():
    """Validate configuration settings"""
    warnings = []
    
    # Check CAPTCHA service availability
    if not any([external_captcha_service_config.TWOCAPTCHA_API_KEY, external_captcha_service_config.ANTICAPTCHA_API_KEY, external_captcha_service_config.CAPMONSTER_API_KEY]):
        warnings.append("No CAPTCHA service API keys configured. CAPTCHA solving will be disabled.")
    
    # Check proxy configuration
    if proxy_config.USE_PROXY and not proxy_config.PROXY_SERVER:
        warnings.append("Proxy is enabled but no proxy server configured.")
    
    # Check browser settings
    if browser_config.DEFAULT_HEADLESS_MODE and anti_detection_config.USE_STEALTH_PLUGIN:
        warnings.append("Stealth mode may be less effective in headless mode.")
    
    return warnings

# Export commonly used configs
__all__ = [
    'anti_detection_config',
    'external_captcha_service_config',
    'browser_config',
    'general_config',
    'proxy_config',
    'captcha_config',
    'security_config',
    'pydantic_settings',
    'mem0_adapter_config',
    'validate_config'
]
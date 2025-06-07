import asyncio
import sys
import base64
import json
import logging
import os
import gc
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from web_automation.vision.image_analyzer import ImageAnalyzer
from urllib.parse import urlparse

from playwright.async_api import (
    async_playwright,
    Page,
    Browser,
    BrowserContext,
    ElementHandle,
    TimeoutError as PlaywrightTimeoutError
)
from playwright_stealth import stealth_async
import uuid
from web_automation.utils.url_helpers import is_similar_url, get_domain
from web_automation.captcha.captcha_handler import CaptchaIntegration
from web_automation.captcha.captcha_handler import VisionCaptchaHandler
from pydantic import ValidationError
# Import moved to factory.py to avoid circular imports
from ..memory.memory_manager import Mem0BrowserAdapter
from ..config import settings
from ..config.settings import mem0_adapter_config  # Import the specific config
from ..models.instructions import (
    InstructionSet,
    BrowserInstruction,
    ActionType,
    WaitCondition,
    ClickInstruction,
    TypeInstruction,
    WaitInstruction,
    ScrollInstruction,
    ScreenshotInstruction,
    ExtractInstruction,
    NavigateInstruction,
    EvaluateInstruction,
    UploadInstruction,
    DownloadInstruction
)

from ..config import settings
from ..utils.fingerprint import (
    create_consistent_fingerprint,
    generate_canvas_noise,
    get_random_user_agent,
    get_random_accept_language,
    get_random_platform,
    get_random_viewport,
    get_random_timezone
)

# Configuration is now accessed via the settings module, e.g., settings.anti_detection_config

logger = logging.getLogger(__name__)

class BrowserAgentError(Exception):
    """Base exception for browser agent errors."""
    pass

class InstructionExecutionError(BrowserAgentError):
    """Raised when an instruction fails to execute."""
    def __init__(self, instruction: Dict, message: str):
        self.instruction = instruction
        self.message = message
        super().__init__(f"Failed to execute instruction {instruction.get('type')}: {message}")

class AgentState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    AWAITING_NAVIGATION = "awaiting_navigation"
    CAPTCHA_REQUIRED = "captcha_required"
    UNEXPECTED_MODAL = "unexpected_modal"
    FATAL_ERROR = "fatal_error"
    RECOVERING = "recovering"

class PlaywrightBrowserAgent:
    """
    A browser automation agent that executes JSON-based instructions using Playwright.
    Provides explicit image analysis API (analyze_image, analyze_current_page_visually, compare_images) via lazy-loaded ImageAnalyzer.
    """
    
    def __init__(self, dependencies: 'BrowserAgentDependencies', **kwargs):
        """Initialize the browser agent with required dependencies.
        
        Args:
            dependencies: BrowserAgentDependencies instance containing required services
            **kwargs: Additional configuration overrides (will override config from dependencies)
        """
        self.deps = dependencies
        
        # Get config from dependencies
        config = dependencies.config
        
        # Set attributes from config, with overrides from kwargs
        self.identity_id = kwargs.get('identity_id') or config.get('identity_id', 'default')
        self.browser_type = (kwargs.get('browser_type') or config.get('browser_type', 'chromium')).lower()
        self.headless = kwargs.get('headless', config.get('headless', True))
        self.stealth = kwargs.get('stealth', config.get('stealth', True))
        self.default_timeout = kwargs.get('default_timeout', config.get('default_timeout', 30000))
        
        # Viewport handling
        default_viewport = {"width": 1920, "height": 1080}
        self.viewport = kwargs.get('viewport') or config.get('viewport', default_viewport)
        
        # User agent and proxy
        self.user_agent = kwargs.get('user_agent') or config.get('user_agent')
        self.proxy = kwargs.get('proxy') or config.get('proxy')
        
        # Paths and directories
        self.downloads_path = kwargs.get('downloads_path') or config.get('downloads_path', 'downloads')
        os.makedirs(self.downloads_path, exist_ok=True)
        
        # Fingerprint settings
        self.randomize_fingerprint = kwargs.get('randomize_fingerprint', 
                                              config.get('randomize_fingerprint', self.stealth))
        self.custom_fingerprint_script = kwargs.get('custom_fingerprint_script') or \
                                       config.get('custom_fingerprint_script')
        
        # Memory management
        self.memory_manager = dependencies.memory_manager
        self.memory_enabled = self.memory_manager is not None
        
        # Initialize browser state
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._fingerprint_profile: Dict = {}
        
        self._reset_execution_state()
        self._fingerprint_profile = self._load_or_create_profile()

        # Initialize executors
        from web_automation.executors.click_executor import ClickExecutor
        from web_automation.executors.type_executor import TypeExecutor
        from web_automation.executors.navigate_executor import NavigateExecutor
        from web_automation.executors.wait_executor import WaitExecutor
        from web_automation.executors.scroll_executor import ScrollExecutor
        from web_automation.executors.screenshot_executor import ScreenshotExecutor
        from web_automation.executors.extract_executor import ExtractExecutor
        from web_automation.executors.evaluate_executor import EvaluateExecutor
        from web_automation.executors.upload_executor import UploadExecutor
        from web_automation.executors.download_executor import DownloadExecutor

        self.executors = {
            ActionType.CLICK: ClickExecutor(),
            ActionType.TYPE: TypeExecutor(),
            ActionType.NAVIGATE: NavigateExecutor(),
            ActionType.WAIT: WaitExecutor(),
            ActionType.SCROLL: ScrollExecutor(),
            ActionType.SCREENSHOT: ScreenshotExecutor(),
            ActionType.EXTRACT: ExtractExecutor(),
            ActionType.EVALUATE: EvaluateExecutor(),
            ActionType.UPLOAD: UploadExecutor(),
            ActionType.DOWNLOAD: DownloadExecutor(),
        }
        logger.info(f"Agent {self.identity_id} initialized with executors")

        # State machine initialization
        self.current_state = AgentState.IDLE
        self.previous_state = AgentState.IDLE
        self._captcha_attempts = 0
        self._max_captcha_attempts = 3

    # --- Explicit Image Analysis API ---
    async def analyze_image(self, image_data: Union[bytes, str], prompt: str = "Describe what you see in this image") -> str:
        """Explicit image analysis always available via lazy-loaded ImageAnalyzer."""
        if not hasattr(self, '_image_analyzer') or self._image_analyzer is None:
            self._image_analyzer = ImageAnalyzer()
        return await self._image_analyzer.analyze_image(image_data, prompt)

    async def analyze_current_page_visually(self, prompt: str = "Analyze this webpage") -> str:
        """Explicit visual analysis of current page via screenshot."""
        screenshot = await self._page.screenshot(type='png', full_page=True)
        return await self.analyze_image(screenshot, prompt)

    async def compare_images(self, image1: Union[bytes, str], image2: Union[bytes, str], comparison_prompt: str = None) -> str:
        """Compare two images for similarities/differences."""
        prompt = comparison_prompt or "Compare these two images and describe the differences"
        if not hasattr(self, '_image_analyzer') or self._image_analyzer is None:
            self._image_analyzer = ImageAnalyzer()
        return await self._image_analyzer.compare_images(image1, image2, prompt)

    def _reset_execution_state(self):
        self._screenshots = []
        self._actions_completed = 0
        self._extracted_data = {}
        self._errors = []
        self._captchas_solved = 0
        self._execution_start_time = None

    def _load_or_create_profile(self) -> Dict:
        # Use deterministic profile names for common scenarios
        if self.identity_id.startswith("test_") or "test" in self.identity_id.lower():
            # Use fixed profile for all tests
            profile_name = "test_profile"
        elif self.identity_id == "default":
            # Use fixed profile for default identity
            profile_name = "default_profile"  
        elif self.identity_id.startswith("production_"):
            # Use fixed profile for production
            profile_name = "production_profile"
        else:
            # Use original identity_id for custom scenarios
            profile_name = self.identity_id
        
        # Create profiles directory if it doesn't exist
        profiles_dir = Path(__file__).resolve().parent.parent / "profiles"
        os.makedirs(profiles_dir, exist_ok=True)
        
        profile_file_path = profiles_dir / f"{profile_name}.json"
        
        # Try to load existing profile
        if profile_file_path.exists():
            logger.info(f"PROFILE: Loading profile '{profile_name}' for identity {self.identity_id} from {profile_file_path}")
            try:
                with open(profile_file_path, 'r') as f:
                    profile = json.load(f)
                # Basic validation or schema check
                if not isinstance(profile, dict) or not profile.get("userAgent"):
                    logger.warning(f"PROFILE: Invalid or empty profile loaded for {profile_name}. Regenerating.")
                    raise FileNotFoundError("Invalid profile format") # Trigger regeneration
                return profile
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"PROFILE: Failed to load or parse profile {profile_name} ({e}). Regenerating.")
        
        # Create new profile if it doesn't exist
        logger.info(f"PROFILE: Creating new profile '{profile_name}' for identity {self.identity_id}")
        
        # Generate a new profile - this will be consistent for the same profile name
        # since we're not using any random values here
        from web_automation.utils.fingerprint import (
            get_random_user_agent,
            get_random_platform,
            get_random_accept_language,
            get_random_timezone,
            get_random_viewport,
            get_random_webgl_info
        )
        
        # Use the profile name as a seed for consistent selection
        random.seed(hash(profile_name) % (2**32))  # Ensure the seed is within int range
        
        # Generate consistent fingerprint components
        user_agent = random.choice([
            ua for ua in [
                # Chrome (Windows)
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                # Chrome (macOS)
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                # Firefox (Windows)
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
                # Firefox (macOS)
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:126.0) Gecko/20100101 Firefox/126.0"
            ]
        ])
        
        # Set platform and timezone based on user agent
        if "Windows" in user_agent:
            platform = "Win32"
            timezone = random.choice(["America/New_York", "America/Los_Angeles"])
        elif "Macintosh" in user_agent:
            platform = "MacIntel"
            timezone = random.choice(["America/New_York", "America/Los_Angeles"])
        else:
            platform = "Linux x86_64"
            timezone = "Europe/London"
        
        # Create the profile
        new_profile = {
            "userAgent": user_agent,
            "platform": platform,
            "languages": ["en-US", "en"],
            "timezone": timezone,
            "viewport": {"width": 1920, "height": 1080},
            "webgl": {
                "vendor": "Google Inc. (NVIDIA)",
                "renderer": "ANGLE (NVIDIA, NVIDIA GeForce RTX 3080 Direct3D11 vs_5_0 ps_5_0, D3D11)"
            }
        }
        
        # Save the new profile
        try:
            with open(profile_file_path, 'w') as f:
                json.dump(new_profile, f, indent=4)
            logger.info(f"PROFILE: Saved new profile '{profile_name}' to {profile_file_path}")
        except IOError as e:
            logger.error(f"PROFILE: Failed to save profile '{profile_name}' to {profile_file_path}: {e}")
            # If saving fails, we still return the generated profile for the current session
            # but it won't be persisted.
        
        return new_profile

    async def _human_like_delay(self, min_ms: Optional[int] = None, max_ms: Optional[int] = None, base_delay_type: str = "before"):
        """Introduces a random delay. Uses config defaults if min/max not provided."""
        if base_delay_type == "before":
            min_d = min_ms if min_ms is not None else settings.anti_detection_config.MIN_DELAY_BEFORE_ACTION
            max_d = max_ms if max_ms is not None else settings.anti_detection_config.MAX_DELAY_BEFORE_ACTION
        elif base_delay_type == "after":
            min_d = min_ms if min_ms is not None else settings.anti_detection_config.MIN_DELAY_AFTER_ACTION
            max_d = max_ms if max_ms is not None else settings.anti_detection_config.MAX_DELAY_AFTER_ACTION
        elif base_delay_type == "typing":
            min_d = min_ms if min_ms is not None else settings.anti_detection_config.MIN_TYPING_DELAY_PER_CHAR
            max_d = max_ms if max_ms is not None else settings.anti_detection_config.MAX_TYPING_DELAY_PER_CHAR
        else: # Default to 'before' action style delays
            min_d = min_ms if min_ms is not None else settings.anti_detection_config.MIN_DELAY_BEFORE_ACTION
            max_d = max_ms if max_ms is not None else settings.anti_detection_config.MAX_DELAY_BEFORE_ACTION

        if min_d == 0 and max_d == 0: # Explicitly no delay
            return
        
        delay_ms = random.randint(min_d, max_d) if min_d < max_d else min_d
        
        if delay_ms > 0:
            logger.debug(f"Applying human-like delay: {delay_ms}ms (type: {base_delay_type})")
            await asyncio.sleep(delay_ms / 1000.0) # asyncio.sleep takes seconds
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def initialize(self):
        self._playwright = await async_playwright().start()

        # Use fingerprint profile for session parameters
        # self._fingerprint_profile is loaded/created in __init__
        session_user_agent = self._fingerprint_profile.get("user_agent", self.user_agent) # Fallback to init arg if key missing
        session_viewport = self._fingerprint_profile.get("viewport", self.viewport)     # Fallback to init arg if key missing
        session_platform = self._fingerprint_profile.get("platform", "Win32") # Default fallback
        session_languages = self._fingerprint_profile.get("languages", ["en-US", "en"]) # Default fallback
        session_timezone = self._fingerprint_profile.get("timezone", "America/New_York") # Default fallback

        # Ensure viewport has width and height, providing defaults if necessary from profile or config
        if not isinstance(session_viewport, dict) or "width" not in session_viewport or "height" not in session_viewport:
            logger.warning(f"PROFILE: Viewport from profile is invalid or incomplete: {session_viewport}. Falling back to default.")
            session_viewport = self.viewport # self.viewport is set from config or init args

        # Ensure languages is a list
        if not isinstance(session_languages, list):
            logger.warning(f"PROFILE: Languages from profile is not a list: {session_languages}. Falling back to default.")
            session_languages = ["en-US", "en"]

        launch_options = {
            "headless": self.headless,
            "timeout": self.default_timeout,
            "args": settings.browser_config.EXTRA_BROWSER_ARGS if hasattr(settings.browser_config, 'EXTRA_BROWSER_ARGS') else []
        }
        if self.proxy:
            launch_options["proxy"] = self.proxy
        
        # Enhanced browser_config options:
        if hasattr(settings.browser_config, 'BROWSER_ARGS'):
            launch_options["args"] = launch_options["args"] + settings.browser_config.BROWSER_ARGS
        
        browser_launcher = getattr(self._playwright, self.browser_type, None)
        if not browser_launcher:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        self._browser = await browser_launcher.launch(**launch_options)
        
        context_options = {
            "viewport": session_viewport,
            "user_agent": session_user_agent,
            "locale": session_languages[0] if session_languages else 'en-US',
            "timezone_id": session_timezone,
            "ignore_https_errors": True,
            "accept_downloads": True,
            # Enhanced browser_config options:
            "java_script_enabled": not getattr(settings.browser_config, 'DISABLE_JAVASCRIPT', False),
            "bypass_csp": True,
        }
        # Store for JS overrides
        # Store for potential JS overrides or logging, sourced from profile
        self._session_platform = session_platform 
        self._session_languages = session_languages
        self._session_timezone = session_timezone
        self._session_user_agent = session_user_agent

        logger.info(f"PROFILE: Initializing context with Profile ID {self.identity_id}:")
        logger.info(f"  User-Agent: {session_user_agent}")
        logger.info(f"  Viewport: {session_viewport}")
        logger.info(f"  Platform: {session_platform}")
        logger.info(f"  Languages: {session_languages}")
        logger.info(f"  Timezone: {session_timezone}")

        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page() # Page created before stealth
        await self._enable_stealth() # Stealth applied to page
    # End of initialize method, ensure proper unindent for next method

    async def close(self):
        logger.info("Attempting to close browser resources...")
        if self._page:
            try:
                logger.info("Closing browser page...")
                await self._page.close()
                self._page = None
                logger.info("Browser page closed.")
            except Exception as e:
                logger.error(f"PAGE_CLOSE_ERROR: Error closing page: {e}", exc_info=True)
        if self._context:
            try:
                logger.info("Closing browser context...")
                await self._context.close()
                self._context = None
                logger.info("Browser context closed.")
            except Exception as e:
                logger.error(f"CONTEXT_CLOSE_ERROR: Error closing context: {e}", exc_info=True)
        if self._browser:
            try:
                logger.info("Closing browser...")
                await self._browser.close()
                self._browser = None
                logger.info("Browser closed.")
            except Exception as e:
                logger.error(f"BROWSER_CLOSE_ERROR: Error closing browser: {e}", exc_info=True)
        if self._playwright:
            try:
                logger.info("Stopping Playwright...")
                await asyncio.sleep(0.1) # Small delay for graceful shutdown
                await self._playwright.stop()
                self._playwright = None
                logger.info("Playwright stopped.")
            except Exception as e:
                logger.error(f"PLAYWRIGHT_STOP_ERROR: Error stopping Playwright: {e}", exc_info=True)
        logger.info("Browser resources closed.")
    
    async def _enable_stealth(self):
        if not self._context or not self._page:
            logger.warning("STEALTH: Context or Page not available, skipping playwright-stealth application.")
            return

        try:
            logger.info("STEALTH: Attempting to apply playwright-stealth protection...")
            await stealth_async(self._page)
            logger.info("STEALTH: Successfully applied playwright-stealth protection.")
        except Exception as e:
            logger.error(f"STEALTH: Error applying playwright-stealth: {e}", exc_info=True)
            # Decide if this should re-raise or if the agent can continue without stealth
            # For now, logging the error and continuing.

        # The base_stealth_script for navigator.webdriver is handled by playwright-stealth
        # If specific other init scripts are still needed, they could be added here.
        # For now, assuming playwright-stealth covers the necessary basics.

        logger.info("STEALTH: _enable_stealth completed using playwright-stealth.")

    async def _apply_memory_context(self, instruction_data, memories: List[Dict]) -> any:
        """
        Dynamically modify instruction based on retrieved memories.
        This is where the magic happens - memories actually improve performance!
        """
        if not memories:
            return instruction_data
        
        # Create a mutable copy of the instruction
        enhanced_instruction = instruction_data.copy() if hasattr(instruction_data, 'copy') else instruction_data
        
        # Extract learned patterns from memories
        for memory in memories:
            memory_text = memory.get('memory', '')
            
            # Speed optimization from memory
            if 'slow clicking preferred' in memory_text.lower():
                if hasattr(enhanced_instruction, 'delay'):
                    enhanced_instruction.delay = max(getattr(enhanced_instruction, 'delay', 0) or 0, 2.0)
                    logger.info(f"Applied memory: increased delay to {enhanced_instruction.delay}s")
            
            # Selector optimization from memory
            if 'successful selector:' in memory_text.lower():
                # Extract successful selector from memory
                success_match = re.search(r'successful selector: ([^\s]+)', memory_text)
                if success_match and hasattr(enhanced_instruction, 'selector'):
                    learned_selector = success_match.group(1)
                    if getattr(enhanced_instruction, 'selector', None) != learned_selector:
                        logger.info(f"Applying learned selector: {learned_selector}")
                        enhanced_instruction.selector = learned_selector
            
            # Wait strategy from memory
            if 'wait strategy:' in memory_text.lower():
                wait_match = re.search(r'wait strategy: (\d+)', memory_text)
                if wait_match:
                    learned_wait = int(wait_match.group(1))
                    if hasattr(enhanced_instruction, 'wait_time'):
                        current_wait = getattr(enhanced_instruction, 'wait_time', 0) or 0
                        enhanced_instruction.wait_time = max(current_wait, learned_wait)
                        logger.info(f"Applied memory: wait time {learned_wait}ms")
        
        return enhanced_instruction

    async def _store_execution_success(self, instruction_data, execution_time: float):
        """Store successful execution patterns for future learning"""
        if not self.memory_manager:
            return
        
        try:
            # Store timing optimization
            instruction_type = getattr(instruction_data, 'type', 'unknown')
            instruction_type_str = instruction_type.value if hasattr(instruction_type, 'value') else str(instruction_type)
            
            timing_pattern = f"Successful timing for {instruction_type_str}: {execution_time:.2f}s"
            
            # Store selector success
            if hasattr(instruction_data, 'selector') and getattr(instruction_data, 'selector', None):
                selector_pattern = f"Successful selector: {instruction_data.selector} for {instruction_type_str}"
                self.memory_manager.store_automation_pattern(
                    pattern=selector_pattern,
                    success=True,
                    user_id=self.identity_id,
                    metadata={"timing": execution_time, "instruction_type": instruction_type_str}
                )
            
            # Store general success pattern
            self.memory_manager.store_automation_pattern(
                pattern=timing_pattern,
                success=True,
                user_id=self.identity_id,
                metadata={"execution_time": execution_time}
            )
        except Exception as e:
            logger.error(f"Error storing execution success: {e}")

    async def _handle_execution_failure(self, instruction_data, error: Exception):
        """Learn from failures and suggest improvements"""
        if not self.memory_manager:
            return
        
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            instruction_type = getattr(instruction_data, 'type', 'unknown')
            instruction_type_str = instruction_type.value if hasattr(instruction_type, 'value') else str(instruction_type)
            
            # Store failure pattern
            failure_pattern = f"Failed {instruction_type_str}: {error_type} - {error_message[:100]}"
            
            # Check for similar past failures
            similar_failures = self.memory_manager.search_memory(
                query=f"Failed {instruction_type_str} {error_type}",
                user_id=self.identity_id,
                limit=3
            )
            
            if len(similar_failures) >= 2:  # Pattern detected
                logger.warning(f"Repeated failure pattern detected for {instruction_type_str}")
                
                # Suggest alternative approach
                suggestion = self._generate_failure_suggestion(instruction_data, similar_failures)
                if suggestion:
                    self.memory_manager.store_automation_pattern(
                        pattern=f"Suggested fix for {instruction_type_str}: {suggestion}",
                        success=True,  # This is a positive suggestion
                        user_id=self.identity_id,
                        metadata={"type": "suggestion", "original_error": error_type}
                    )
            
            # Store this failure
            self.memory_manager.store_automation_pattern(
                pattern=failure_pattern,
                success=False,
                user_id=self.identity_id,
                metadata={"error_type": error_type}
            )
        except Exception as e:
            logger.error(f"Error handling execution failure: {e}")

    def _generate_failure_suggestion(self, instruction_data, similar_failures: List[Dict]) -> str:
        """Generate actionable suggestions based on failure patterns"""
        try:
            error_types = [f.get('metadata', {}).get('error_type', '') for f in similar_failures]
            
            if 'TimeoutError' in error_types:
                return "increase wait time or use explicit wait conditions"
            elif 'ElementNotFoundError' in error_types:
                return "try alternative selector strategy or wait for element visibility"
            elif 'StaleElementReferenceError' in error_types:
                return "re-find element before interaction"
            else:
                return "consider adding retry logic or human-like delays"
        except Exception:
            return "review automation strategy"

    async def _execute_instruction_with_memory(self, instruction_data, user_id: str) -> None:
        """Memory-enhanced instruction execution with dynamic optimization"""
        memories = []
        if self.memory_manager:
            try:
                # Search for relevant patterns
                instruction_type = getattr(instruction_data, 'type', 'unknown')
                instruction_type_str = instruction_type.value if hasattr(instruction_type, 'value') else str(instruction_type)
                
                search_query = f"{instruction_type_str}"
                if hasattr(instruction_data, 'selector') and getattr(instruction_data, 'selector', None):
                    search_query += f" {instruction_data.selector}"
                
                memories = self.memory_manager.search_automation_patterns(
                    pattern_query=search_query,
                    user_id=user_id,
                    limit=5
                )
                
                if memories:
                    logger.info(f"Found {len(memories)} relevant memories for {instruction_type_str}")
            except Exception as e:
                logger.error(f"Error searching automation patterns: {e}")
        
        # Apply memory insights to instruction
        enhanced_instruction = await self._apply_memory_context(instruction_data, memories)
        
        action_type = getattr(enhanced_instruction, 'type', getattr(instruction_data, 'type'))
        executor = self.executors.get(action_type)

        if not executor:
            error_msg = f"No executor found for action type '{action_type.value if hasattr(action_type, 'value') else action_type}'"
            logger.error(error_msg)
            # Create a simple exception for _handle_execution_failure
            temp_exception = Exception(error_msg)
            await self._handle_execution_failure(enhanced_instruction, temp_exception)
            self._errors.append({
                "type": "UnsupportedInstructionError",
                "instruction_type": action_type.value if hasattr(action_type, 'value') else str(action_type),
                "message": error_msg
            })
            # Serialize instruction for InstructionExecutionError
            instruction_dict = {}
            if hasattr(enhanced_instruction, 'model_dump'): instruction_dict = enhanced_instruction.model_dump()
            elif hasattr(enhanced_instruction, 'dict'): instruction_dict = enhanced_instruction.dict()
            else:
                try: instruction_dict = vars(enhanced_instruction)
                except TypeError: instruction_dict = {"type": str(action_type), "error": "Could not serialize instruction"}
            raise InstructionExecutionError(instruction_dict, error_msg)

        # --- Retry Logic Start ---
        total_attempts = getattr(enhanced_instruction, 'retry_attempts', 1)
        retry_delay_ms = getattr(enhanced_instruction, 'retry_delay', 1000)

        if not isinstance(total_attempts, int) or total_attempts < 1:
            logger.warning(f"Invalid retry_attempts ({total_attempts}) for {action_type}. Defaulting to 1.")
            total_attempts = 1
        if not isinstance(retry_delay_ms, (int, float)) or retry_delay_ms < 0:
            logger.warning(f"Invalid retry_delay_ms ({retry_delay_ms}) for {action_type}. Defaulting to 1000ms.")
            retry_delay_ms = 1000

        last_exception = None
        start_time_instruction = time.time() # For overall instruction timing including retries

        if action_type not in [ActionType.WAIT, ActionType.EVALUATE]:
            await self._human_like_delay(base_delay_type="before")

        for attempt in range(total_attempts):
            try:
                logger.info(f"Attempt {attempt + 1}/{total_attempts} for instruction: {action_type.value if hasattr(action_type, 'value') else str(action_type)} - Selector: {getattr(enhanced_instruction, 'selector', 'N/A')}")
                
                await executor.execute(self._page, enhanced_instruction) # Core action execution
                
                self._actions_completed += 1
                
                if action_type not in [ActionType.WAIT, ActionType.EVALUATE]:
                    await self._human_like_delay(base_delay_type="after")
                
                execution_time = time.time() - start_time_instruction
                await self._store_execution_success(enhanced_instruction, execution_time)
                
                action_type_str = action_type.value if hasattr(action_type, 'value') else str(action_type)
                logger.debug(f"Successfully executed {action_type_str} in {execution_time:.2f}s (attempt {attempt + 1}/{total_attempts}) with memory enhancement")
                
                last_exception = None 
                break 

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1}/{total_attempts} failed for {action_type.value if hasattr(action_type, 'value') else str(action_type)} ({getattr(enhanced_instruction, 'selector', 'N/A')}): {e}")
                if attempt < total_attempts - 1:
                    logger.info(f"Waiting {retry_delay_ms / 1000:.2f}s before next attempt...")
                    await asyncio.sleep(retry_delay_ms / 1000)
        # --- Retry Logic End ---

        if last_exception:
            logger.error(f"All {total_attempts} attempts failed for {action_type.value if hasattr(action_type, 'value') else str(action_type)} ({getattr(enhanced_instruction, 'selector', 'N/A')}). Last error: {last_exception}", exc_info=True)
            await self._handle_execution_failure(enhanced_instruction, last_exception)
            
            error_type_name = type(last_exception).__name__
            error_message_str = str(last_exception)

            if isinstance(last_exception, InstructionExecutionError):
                error_instruction_dict = last_exception.instruction if isinstance(last_exception.instruction, dict) else {}
                self._errors.append({
                    "type": "InstructionExecutionError", 
                    "instruction_type": error_instruction_dict.get('type', str(action_type)),
                    "selector": error_instruction_dict.get('selector', getattr(enhanced_instruction, 'selector', 'N/A')),
                    "message": last_exception.message
                })
            else:
                self._errors.append({
                    "type": error_type_name,
                    "instruction_type": action_type.value if hasattr(action_type, 'value') else str(action_type),
                    "selector": getattr(enhanced_instruction, 'selector', 'N/A'),
                    "message": error_message_str
                })
            raise last_exception

    async def _check_page_state(self) -> AgentState:
        """
        Check the current state of the page to detect unexpected conditions like CAPTCHA or modals.

        Returns:
            AgentState: The detected state of the agent.
        """
        if not self._page:
            logger.error("Cannot check page state: Page not initialized")
            return AgentState.FATAL_ERROR

        try:
            # Check for CAPTCHA
            captcha_indicators = [
                "#g-recaptcha", 
                ".g-recaptcha", 
                "#recaptcha", 
                "[data-sitekey]", 
                "iframe[src*='recaptcha']",
                "div:contains('CAPTCHA')",
                "div:contains('captcha')"
            ]
            for indicator in captcha_indicators:
                element = await self._page.query_selector(indicator)
                if element and await element.is_visible():
                    logger.warning(f"CAPTCHA detected with selector: {indicator}")
                    return AgentState.CAPTCHA_REQUIRED

            # Check for unexpected modals or pop-ups (adjust selectors based on common patterns)
            modal_indicators = [
                "div.modal", 
                "div.popup", 
                "#overlay", 
                "div:contains('login')", 
                "div:contains('sign in')",
                "div:contains('session expired')"
            ]
            for indicator in modal_indicators:
                element = await self._page.query_selector(indicator)
                if element and await element.is_visible():
                    logger.warning(f"Unexpected modal detected with selector: {indicator}")
                    return AgentState.UNEXPECTED_MODAL

            # Check URL for unexpected navigation (like a login page or error page)
            current_url = self._page.url
            if 'login' in current_url.lower() or 'signin' in current_url.lower():
                logger.warning(f"Unexpected navigation to login page: {current_url}")
                return AgentState.UNEXPECTED_MODAL

            # If no issues detected, return to normal state
            return AgentState.EXECUTING if self.current_state != AgentState.IDLE else AgentState.IDLE
        except Exception as e:
            logger.error(f"Error checking page state: {e}", exc_info=True)
            return AgentState.FATAL_ERROR

    async def _handle_state_transition(self, new_state: AgentState) -> bool:
        """
        Handle transition to a new state and execute appropriate recovery actions.

        Args:
            new_state: The new state to transition to.

        Returns:
            bool: True if the state was handled successfully, False otherwise.
        """
        self.previous_state = self.current_state
        self.current_state = new_state
        logger.info(f"State transition: {self.previous_state.value} -> {self.current_state.value}")

        if new_state == AgentState.CAPTCHA_REQUIRED:
            return await self._handle_captcha_state()
        elif new_state == AgentState.UNEXPECTED_MODAL:
            return await self._handle_modal_state()
        elif new_state == AgentState.FATAL_ERROR:
            logger.error("Fatal error state reached. Stopping execution.")
            return False
        elif new_state in [AgentState.IDLE, AgentState.EXECUTING, AgentState.AWAITING_NAVIGATION]:
            return True  # Normal states, no special handling needed
        elif new_state == AgentState.RECOVERING:
            logger.info("Agent is in recovering state")
            return True
        else:
            logger.warning(f"Unhandled state: {new_state.value}")
            return False

    async def _handle_captcha_state(self) -> bool:
        """
        Handle the CAPTCHA_REQUIRED state by attempting to solve the CAPTCHA.

        Returns:
            bool: True if CAPTCHA was solved or max attempts reached, False if a fatal error occurs.
        """
        if self._captcha_attempts >= self._max_captcha_attempts:
            logger.error(f"Max CAPTCHA attempts ({self._max_captcha_attempts}) reached. Cannot solve.")
            self._captcha_attempts = 0  # Reset for next time
            self.current_state = AgentState.FATAL_ERROR
            return False

        self._captcha_attempts += 1
        logger.info(f"Attempting to solve CAPTCHA (Attempt {self._captcha_attempts}/{self._max_captcha_attempts})")
        
        if 'captcha' in self.plugins:
            try:
                success = await self.plugins['captcha'].solve()
                if success:
                    logger.info("CAPTCHA solved successfully")
                    self._captcha_attempts = 0  # Reset on success
                    self.current_state = AgentState.RECOVERING
                    # Wait briefly to ensure page updates after CAPTCHA solve
                    await asyncio.sleep(2)
                    return True
                else:
                    logger.warning("Failed to solve CAPTCHA")
                    return False
            except Exception as e:
                logger.error(f"Error solving CAPTCHA: {e}", exc_info=True)
                return False
        else:
            logger.error("CAPTCHA plugin not available. Cannot solve CAPTCHA.")
            self.current_state = AgentState.FATAL_ERROR
            return False

    async def _handle_modal_state(self) -> bool:
        """
        Handle the UNEXPECTED_MODAL state by attempting to dismiss or handle the modal.

        Returns:
            bool: True if modal was handled, False otherwise.
        """
        logger.info("Handling unexpected modal or login page")
        # Simple strategy: Try to find and click a close button
        close_button_selectors = [
            "button:contains('close')",
            "button:contains('dismiss')",
            "button:contains('X')",
            "a:contains('close')",
            "#close",
            ".close",
            ".modal-close"
        ]

        for selector in close_button_selectors:
            try:
                element = await self._page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    logger.info(f"Clicked close button with selector: {selector}")
                    await asyncio.sleep(1)  # Wait for modal to disappear
                    self.current_state = AgentState.RECOVERING
                    return True
            except Exception as e:
                logger.error(f"Error trying to close modal with {selector}: {e}")

        logger.warning("Could not handle unexpected modal. No close button found.")
        return False

    async def execute_instruction_with_state_management(self, instruction: Any) -> Any:
        """
        Execute a single instruction with state management to handle unexpected states.

        Args:
            instruction: The instruction to execute.

        Returns:
            Any: Result of the instruction execution.
        """
        if not self._page:
            raise RuntimeError("Browser page not initialized. Call start() first.")

        # Check initial state before execution
        detected_state = await self._check_page_state()
        if detected_state != self.current_state and detected_state not in [AgentState.IDLE, AgentState.EXECUTING]:
            success = await self._handle_state_transition(detected_state)
            if not success and detected_state == AgentState.FATAL_ERROR:
                raise RuntimeError("Fatal error detected before executing instruction")

        # If in a normal state, proceed with execution
        if self.current_state in [AgentState.IDLE, AgentState.EXECUTING]:
            action_type = instruction.action_type if hasattr(instruction, 'action_type') else instruction.get('action_type', None)
            if not action_type:
                raise ValueError("Instruction does not have an action type")

            executor = self.executors.get(action_type)
            if not executor:
                error_msg = f"No executor found for action type '{action_type.value if hasattr(action_type, 'value') else action_type}'"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"Executing instruction with action type: {action_type}")
            try:
                result = await executor.execute(self._page, instruction)
                logger.info(f"Successfully executed instruction: {action_type}")
                self._actions_completed += 1

                # Check state after execution to catch any immediate issues
                post_execution_state = await self._check_page_state()
                if post_execution_state not in [AgentState.IDLE, AgentState.EXECUTING]:
                    await self._handle_state_transition(post_execution_state)

                return result
            except Exception as e:
                logger.error(f"Error executing instruction {action_type}: {e}", exc_info=True)
                raise
        else:
            logger.warning(f"Skipping instruction execution due to current state: {self.current_state.value}")
            return None

    async def execute_instructions(self, instructions: List[Any]) -> List[Any]:
        """
        Execute a list of instructions with state management.

        Args:
            instructions: List of instructions to execute.

        Returns:
            List[Any]: List of results from each instruction execution.
        """
        results = []
        for instruction in instructions:
            try:
                result = await self.execute_instruction_with_state_management(instruction)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute instruction: {e}", exc_info=True)
                self._errors.append({"type": "ExecutionError", "message": str(e)})
                results.append(None)
                if self.current_state == AgentState.FATAL_ERROR:
                    logger.error("Stopping execution due to fatal error")
                    break
        return results

    # --- Existing Methods (unchanged, for context) ---
    async def start(self) -> None:
        """
        Start the browser agent, initializing Playwright and browser context.
        """
        logger.info(f"Starting browser agent {self.identity_id}")
        self._playwright = await async_playwright().start()
        browser_type = self._playwright.chromium  # Default to chromium
        self._browser = await browser_type.launch(headless=self.config.get('headless', True))
        self._context = await self._browser.new_context()
        self._page = await self._context.new_page()
        self.current_state = AgentState.IDLE
        logger.info(f"Browser agent {self.identity_id} started")

    async def shutdown(self) -> None:
        """
        Shutdown the browser agent, closing all connections.
        """
        logger.info(f"Shutting down browser agent {self.identity_id}")
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self.current_state = AgentState.IDLE
        logger.info(f"Browser agent {self.identity_id} shut down")

    # --- Other methods remain unchanged ---

    async def take_screenshot(self, full_page=False, filename: Optional[str] = None) -> str:
        """Capture screenshot and return base64 or file path."""
        instruction = ScreenshotInstruction(
            type=ActionType.SCREENSHOT, 
            full_page=full_page, 
            filename=filename,
            save_to_disk=bool(filename), # Save if filename is provided
            return_as_base64=not bool(filename) # Return base64 if no filename
        )
        return await self.executors[ActionType.SCREENSHOT].execute(self._page, instruction) or ""

    async def extract_page_data(self, selectors: List[str]) -> Dict[str, Any]:
        """Extract data from specified page elements."""
        extracted_results = {}
        for selector in selectors:
            instruction = ExtractInstruction(type=ActionType.EXTRACT, selector=selector, multiple=False) # Assuming single extract per selector
            try:
                value = await self.executors[ActionType.EXTRACT].execute(self._page, instruction)
                extracted_results[selector] = value
            except Exception as e:
                logger.error(f"Error extracting data for selector {selector}: {e}")
                extracted_results[selector] = {"error": str(e)}
        return extracted_results

    async def get_automation_insights(self, user_id: str = None) -> Dict[str, Any]:
        """Get insights and suggestions based on accumulated memory"""
        if not self.memory_manager:
            return {"insights": [], "suggestions": []}
        
        target_user = user_id or self.identity_id
        
        try:
            # Get recent automation patterns
            recent_patterns = self.memory_manager.search_memory(
                query="automation pattern",
                user_id=target_user,
                limit=20
            )
            
            insights = []
            suggestions = []
            
            # Analyze success/failure rates
            successes = [p for p in recent_patterns if p.get('metadata', {}).get('success', False)]
            failures = [p for p in recent_patterns if not p.get('metadata', {}).get('success', True)]
            
            if len(recent_patterns) > 5:
                success_rate = len(successes) / len(recent_patterns) * 100
                insights.append(f"Current automation success rate: {success_rate:.1f}%")
                
                if success_rate < 80:
                    suggestions.append("Consider reviewing failed patterns and adjusting selectors or timing")
            
            # Analyze timing patterns
            execution_times = [
                p.get('metadata', {}).get('execution_time', 0) 
                for p in successes 
                if p.get('metadata', {}).get('execution_time')
            ]
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                insights.append(f"Average execution time: {avg_time:.2f}s")
                
                if avg_time > 5.0:
                    suggestions.append("Automation is running slowly - consider optimizing selectors or reducing delays")
            
            return {
                "insights": insights,
                "suggestions": suggestions,
                "total_patterns": len(recent_patterns),
                "success_rate": len(successes) / max(len(recent_patterns), 1) * 100
            }
        except Exception as e:
            logger.error(f"Error getting automation insights: {e}")
            return {"insights": [], "suggestions": [], "error": str(e)}

    async def solve_captcha(self, captcha_type: str) -> bool:
        """Handle various CAPTCHA types (Placeholder)."""
        # This is a placeholder for CAPTCHA solving integration.
        # Actual implementation would involve calling a CAPTCHA service.
        logger.warning(f"CAPTCHA solving for type '{captcha_type}' is not yet implemented.")
        # Simulate a failed attempt for now
        self._errors.append({"type": "CaptchaError", "message": f"CAPTCHA type '{captcha_type}' not solvable."})
        return False


# Apply CAPTCHA integration methods to the WebBrowserAgent class
PlaywrightBrowserAgent = CaptchaIntegration.add_captcha_methods(PlaywrightBrowserAgent)

async def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    logger.info("Initializing PlaywrightBrowserAgent for CAPTCHA test...")
    agent = PlaywrightBrowserAgent(headless=False, stealth=True) # Ensure headless is False for visibility
    captcha_handler = VisionCaptchaHandler() # Initialize the CAPTCHA handler

    async with agent:
        logger.info("PlaywrightBrowserAgent entered context. Navigating to reCAPTCHA test page...")
        # Navigate to a reCAPTCHA v2 test page
        # You might need to find a reliable public test page, or set up your own.
        # For demonstration, we'll use a common one, but it might change or have rate limits.
        recaptcha_test_url = "https://www.google.com/recaptcha/api2/demo"
        await agent._page.goto(recaptcha_test_url, wait_until="domcontentloaded")
        logger.info(f"Navigated to: {recaptcha_test_url}")

        logger.info("Attempting to handle CAPTCHA...")
        # Use the captcha_handler to detect and solve CAPTCHAs on the current page
        captcha_solved = await captcha_handler.handle_page_captchas(agent._page)
        
        if captcha_solved:
            logger.info("CAPTCHA successfully handled!")
        else:
            logger.warning("Failed to handle CAPTCHA.")

        logger.info("CAPTCHA handling complete. Pausing for human inspection (30 seconds)...")
        logger.info("You can inspect the browser window to see the result.")
        await asyncio.sleep(30) # Pause for human inspection
        logger.info(f"CAPTCHA Handler Stats: {captcha_handler.get_stats()}")

    logger.info("PlaywrightBrowserAgent exited context. Test finished.")


if __name__ == "__main__":
    # Setup logging
    log_file_path = Path(settings.general_config.LOGS_DIR) / "browser_agent.log"
    log_level = getattr(logging, settings.general_config.LOG_LEVEL.upper(), logging.INFO)

    # Basic configuration for the root logger
    logging.basicConfig(level=log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[]) # Start with no handlers, add them below

    # Get the root logger
    root_logger = logging.getLogger('') 
    # Or, get the specific logger for this module if preferred: 
    # script_logger = logging.getLogger(__name__) # and then use script_logger.addHandler

    # Add StreamHandler to output to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    if settings.general_config.LOG_TO_FILE:
        # Ensure logs directory exists (settings.py should also do this)
        Path(settings.general_config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {log_file_path}")
    else:
        root_logger.info("File logging is disabled in settings.")

    root_logger.info("Logging setup complete.")
    
    asyncio.run(main())

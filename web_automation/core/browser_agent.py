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
from ..config.settings import mem0_adapter_config, reasoning_config # Import the specific config
from ..reasoning.reasoning_engine import WebAutomationReasoner
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

class PlaywrightBrowserAgent:
    """
    A browser automation agent that executes JSON-based instructions using Playwright.
    """
    
    def __init__(self, dependencies: Optional['BrowserAgentDependencies'] = None, **kwargs):
        # Handle both new DI pattern and legacy pattern
        if dependencies is not None:
            # New DI pattern
            self.deps = dependencies
            self.identity_id = dependencies.config.get('identity_id', 'default')
            self.memory_manager = dependencies.memory_manager
            
            # Extract config
            config = dependencies.config
            self.browser_type = config.get('browser_type', 'chromium')
            self.headless = config.get('headless', True)
            self.stealth = config.get('stealth', True)
            self.default_timeout = config.get('default_timeout', 30000)
            self.viewport = config.get('viewport', {"width": 1920, "height": 1080})
            self.user_agent = config.get('user_agent')
            self.proxy = config.get('proxy')
            self.downloads_path = config.get('downloads_path', 'downloads')
            self.randomize_fingerprint = config.get('randomize_fingerprint', True)
            self.custom_fingerprint_script = config.get('custom_fingerprint_script')
            self.memory_enabled = self.memory_manager is not None
        else:
            # Legacy pattern - keep all existing logic exactly as is
            self.identity_id = kwargs.get('identity_id') or str(uuid.uuid4())
            self.browser_type = (kwargs.get('browser_type') or settings.browser_config.DEFAULT_BROWSER_TYPE).lower()
            self.headless = kwargs.get('headless', settings.browser_config.DEFAULT_HEADLESS_MODE)
            self.stealth = kwargs.get('stealth', True)
            self.default_timeout = kwargs.get('default_timeout') or settings.general_config.DEFAULT_TIMEOUT
            self.viewport = kwargs.get('viewport') or {"width": settings.browser_config.DEFAULT_VIEWPORT_WIDTH, "height": settings.browser_config.DEFAULT_VIEWPORT_HEIGHT}
            self.user_agent = kwargs.get('user_agent')
            
            proxy = kwargs.get('proxy')
            if proxy is not None:
                self.proxy = proxy
            elif settings.proxy_config.USE_PROXY and settings.proxy_config.PROXY_SERVER:
                self.proxy = {"server": settings.proxy_config.PROXY_SERVER}
                if settings.proxy_config.PROXY_USERNAME and settings.proxy_config.PROXY_PASSWORD:
                    self.proxy["username"] = settings.proxy_config.PROXY_USERNAME
                    self.proxy["password"] = settings.proxy_config.PROXY_PASSWORD
            else:
                self.proxy = None
                
            self.downloads_path = kwargs.get('downloads_path') or settings.general_config.DOWNLOADS_DIR
            os.makedirs(self.downloads_path, exist_ok=True)
            
            self.randomize_fingerprint = kwargs.get('randomize_fingerprint', self.stealth)
            self.custom_fingerprint_script = kwargs.get('custom_fingerprint_script')
            self.memory_enabled = kwargs.get('memory_enabled', True)
            
            # Initialize memory manager the old way
            self.memory_manager: Optional[Mem0BrowserAdapter] = None
            if self.memory_enabled:
                try:
                    self.memory_manager = Mem0BrowserAdapter(mem0_config=mem0_adapter_config)
                except Exception as e:
                    logger.error(f"Failed to initialize Mem0BrowserAdapter: {e}")
                    self.memory_manager = None
        
        # Common initialization for both patterns
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._fingerprint_profile: Dict = {}
        
        self._reset_execution_state()
        self._fingerprint_profile = self._load_or_create_profile()
        self._init_reasoning_engine() # Initialize reasoning engine

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
        handler = self._get_action_handler(action_type)

        if not handler:
            error_msg = f"No handler found for action type '{action_type.value if hasattr(action_type, 'value') else action_type}'"
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
                
                await handler(enhanced_instruction) # Core action execution
                
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

    async def execute_instructions(self, instructions_json: Union[Dict, InstructionSet]) -> Dict:
        self._reset_execution_state()
        self._execution_start_time = time.time()
        current_session_context = {"initial_url": None, "history": []}

        if isinstance(instructions_json, dict):
            try:
                instruction_set = InstructionSet(**instructions_json)
            except ValidationError as e:
                self._errors.append({"type": "ValidationError", "message": str(e)})
                return self._prepare_result(success=False)
        elif isinstance(instructions_json, InstructionSet):
            instruction_set = instructions_json
        else:
            self._errors.append({"type": "InvalidInput", "message": "instructions_json must be a dict or InstructionSet"})
            return self._prepare_result(success=False)

        if not self._page:
            self._errors.append({"type": "BrowserError", "message": "Page not initialized."})
            return self._prepare_result(success=False)

        if self.memory_manager:
            logger.info(f"MEMORY: Retrieving session context for user {self.identity_id}")
            retrieved_contexts = await self.memory_manager.search_session_context(
                user_id=self.identity_id,
                query="last known state or initial context", # General query
                limit=1 
            )
            if retrieved_contexts and retrieved_contexts[0].get("data"):
                # Ensure data is a dict
                retrieved_data = retrieved_contexts[0]["data"]
                if isinstance(retrieved_data, str):
                    try:
                        current_session_context.update(json.loads(retrieved_data))
                    except json.JSONDecodeError:
                         logger.warning(f"MEMORY: Failed to parse retrieved session context string: {retrieved_data}")
                         current_session_context.update({"raw_retrieved_context": retrieved_data}) # store as is
                elif isinstance(retrieved_data, dict):
                     current_session_context.update(retrieved_data)
                else:
                    logger.warning(f"MEMORY: Retrieved session context data is not a dict or parsable string: {type(retrieved_data)}")
                    current_session_context.update({"raw_retrieved_context": str(retrieved_data)})

                logger.info(f"MEMORY: Retrieved and updated session context: {current_session_context}")
        
        current_session_context["initial_url"] = instruction_set.url or (self._page.url if self._page else None)

        if instruction_set.url:
            nav_instruction = NavigateInstruction(type=ActionType.NAVIGATE, url=instruction_set.url)
            logger.info(f"Executing initial navigation to: {instruction_set.url}")
            success = await self._execute_instruction_with_memory(nav_instruction, current_session_context)
            if not success:
                return self._prepare_result(success=False)
            current_session_context["history"].append({
                "action": ActionType.NAVIGATE.value, 
                "url": instruction_set.url, 
                "status": "success",
                "timestamp": time.time()
            })
            current_session_context["last_known_url"] = self._page.url if self._page else instruction_set.url

        for instruction_data in instruction_set.instructions:
            logger.info(f"Executing instruction: {instruction_data.type.value} - Selector: {getattr(instruction_data, 'selector', 'N/A')}")
            success = await self._execute_instruction_with_memory(instruction_data, current_session_context)
            
            action_details_for_history = instruction_data.dict()
            if 'type' in action_details_for_history and isinstance(action_details_for_history['type'], Enum):
                 action_details_for_history['type'] = action_details_for_history['type'].value

            current_session_context["history"].append({
                "instruction": action_details_for_history,
                "status": "success" if success else "failure",
                "timestamp": time.time(),
                "current_url": self._page.url if self._page else current_session_context.get("last_known_url")
            })
            
            if not success:
                return self._prepare_result(success=False) # Stop on first error
            
            current_session_context["last_known_url"] = self._page.url if self._page else current_session_context.get("last_known_url")
            current_session_context["last_successful_action"] = action_details_for_history

        if self.memory_manager and not self._errors:
            final_context_to_store = {
                "last_known_url": self._page.url if self._page else current_session_context.get("last_known_url"),
                "actions_completed_in_run": self._actions_completed,
                "total_instructions_in_set": len(instruction_set.instructions) + (1 if instruction_set.url else 0),
                "final_status": "success",
                "extracted_data_summary": list(self._extracted_data.keys()),
                "history_summary": f"{len(current_session_context.get('history', []))} steps in this run.",
                "timestamp": time.time()
            }
            logger.info(f"MEMORY: Storing final session context for user {self.identity_id}")
            await self.memory_manager.store_session_context(
                user_id=self.identity_id,
                context_data=final_context_to_store 
            )
        
        return self._prepare_result(success=True and not self._errors)

    def _prepare_result(self, success: bool) -> Dict:
        execution_time = time.time() - self._execution_start_time
        return {
            "success": success and not self._errors,
            "execution_time": round(execution_time, 2),
            "actions_completed": self._actions_completed,
            "screenshots": self._screenshots,
            "extracted_data": self._extracted_data,
            "errors": self._errors,
            "captchas_solved": self._captchas_solved,
            "final_url": self._page.url if self._page else None
        }

    def _get_action_handler(self, action_type: ActionType) -> Optional[Callable]:
        """Returns the appropriate handler function for a given action type."""
        handler_map = {
            ActionType.CLICK: self._handle_click,
            ActionType.TYPE: self._handle_type,
            ActionType.WAIT: self._handle_wait,
            ActionType.SCROLL: self._handle_scroll,
            ActionType.SCREENSHOT: self._handle_screenshot,
            ActionType.EXTRACT: self._handle_extract,
            ActionType.NAVIGATE: self._handle_navigate,
            ActionType.EVALUATE: self._handle_evaluate,
            ActionType.UPLOAD: self._handle_upload,
            ActionType.DOWNLOAD: self._handle_download,
        }
        return handler_map.get(action_type)

    async def _get_element(self, selector: str, timeout: Optional[int] = None) -> ElementHandle:
        if not self._page: raise BrowserAgentError("Page not available")
        try:
            element = await self._page.wait_for_selector(
                selector, 
                state="visible", 
                timeout=timeout or self.default_timeout
            )
            if not element:
                raise InstructionExecutionError({"selector": selector}, f"Element not found or not visible: {selector}")
            return element
        except PlaywrightTimeoutError:
            raise InstructionExecutionError({"selector": selector}, f"Timeout waiting for element: {selector}")

    async def _get_page_or_fail(self) -> "Page":
        """Helper method to get page with proper error handling."""
        if not self._page:
            raise BrowserAgentError("Page not initialized")
        return self._page

    async def _handle_click(self, instruction: ClickInstruction):
        element = await self._get_element(instruction.selector, instruction.timeout)
        await asyncio.sleep(random.uniform(0.1, 0.3)) # Human-like delay
        await element.click(
            button=instruction.button, 
            click_count=instruction.click_count, 
            delay=instruction.delay
        )
        if instruction.wait_for == WaitCondition.NAVIGATION:
            await self._page.wait_for_load_state("load", timeout=instruction.timeout or self.default_timeout)
        elif instruction.wait_for == WaitCondition.NETWORK_IDLE:
            await self._page.wait_for_load_state("networkidle", timeout=instruction.timeout or self.default_timeout)

    async def _handle_type(self, instruction: TypeInstruction):
        element = await self._get_element(instruction.selector, instruction.timeout)
        if instruction.clear:
            await element.fill("")
            await self._human_like_delay(min_ms=50, max_ms=100, base_delay_type="after") # Small delay after clearing

        await element.click() # Focus element before typing
        await self._human_like_delay(min_ms=50, max_ms=150, base_delay_type="before") # Delay after click, before typing first char
        
        for char_to_type in instruction.text:
            await self._get_page_or_fail().keyboard.type(char_to_type)
            # Use explicit delay from instruction if provided, otherwise use configured random typing delay
            char_delay_ms = instruction.delay_between_chars
            if char_delay_ms is not None and char_delay_ms > 0:
                 await asyncio.sleep(char_delay_ms / 1000.0)
            elif char_delay_ms is None: # Only use anti_detection_config if instruction doesn't specify a delay
                await self._human_like_delay(base_delay_type="typing")
            # No delay if instruction.delay_between_chars is 0

    async def _handle_wait(self, instruction: WaitInstruction):
        if not self._page: raise BrowserAgentError("Page not available")
        timeout = instruction.timeout or self.default_timeout
        if instruction.condition == WaitCondition.NAVIGATION:
            await self._page.wait_for_load_state("load", timeout=timeout)
        elif instruction.condition == WaitCondition.NETWORK_IDLE:
            await self._page.wait_for_load_state("networkidle", timeout=timeout)
        elif instruction.condition == WaitCondition.ELEMENT_VISIBLE and instruction.selector:
            await self._get_element(instruction.selector, timeout)
        elif instruction.condition == WaitCondition.ELEMENT_HIDDEN and instruction.selector:
            await self._page.wait_for_selector(instruction.selector, state="hidden", timeout=timeout)
        elif instruction.condition == WaitCondition.TIMEOUT and isinstance(instruction.wait_for, int):
            await asyncio.sleep(instruction.wait_for / 1000)
        else:
            logger.warning(f"Unsupported or misconfigured wait instruction: {instruction.condition}")

    async def _handle_scroll(self, instruction: ScrollInstruction):
        if not self._page: raise BrowserAgentError("Page not available")
        if instruction.scroll_into_view and instruction.selector:
            element = await self._get_element(instruction.selector, instruction.timeout)
            await element.scroll_into_view_if_needed(timeout=instruction.timeout or self.default_timeout)
        elif instruction.x is not None or instruction.y is not None:
            script = f"window.scrollBy({{ left: {instruction.x or 0}, top: {instruction.y or 0}, behavior: '{instruction.behavior}' }})"
            await self._page.evaluate(script)
        await asyncio.sleep(random.uniform(0.2, 0.5)) # Wait for scroll to take effect

    async def _handle_screenshot(self, instruction: ScreenshotInstruction):
        if not self._page:
            raise BrowserAgentError("Page is not initialized. Cannot take screenshot.")
        
        logger.debug(f"_handle_screenshot received instruction.filename: {instruction.filename}")
        filename = instruction.filename
        if not filename: # Generate a default filename if not provided
            filename = f"screenshot_{time.time()}.png"
            logger.debug(f"instruction.filename was empty, generated default filename: {filename}")
        
        # Ensure the screenshots directory exists
        screenshots_dir = Path(general_config.SCREENSHOTS_DIR)
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        full_path = screenshots_dir / filename
        logger.debug(f"_handle_screenshot full_path for screenshot: {full_path}")
        
        try:
            await self._human_like_delay(base_delay_type="before")
            await self._page.screenshot(path=full_path, full_page=instruction.full_page)
            logger.info(f"Screenshot taken: {full_path}")
            self._screenshots.append(str(full_path))
            await self._human_like_delay(base_delay_type="after")
            return {"status": "success", "path": str(full_path)}
        except PlaywrightTimeoutError:
            logger.error(f"Timeout while taking screenshot: {filename}")
            self._errors.append({"type": "ScreenshotError", "message": f"Timeout taking screenshot {filename}"})
            return {"status": "error", "message": "Timeout while taking screenshot"}
        except Exception as e:
            logger.error(f"Error taking screenshot {filename}: {e}")
            self._errors.append({"type": "ScreenshotError", "message": f"Error taking screenshot {filename}: {e}"})
            return {"status": "error", "message": str(e)}

    async def _handle_extract(self, instruction: ExtractInstruction) -> Union[str, List[str], None]:
        if not self._page: raise BrowserAgentError("Page not available")
        if not instruction.selector: 
            raise InstructionExecutionError(instruction.dict(), "Selector is required for extract instruction.")

        elements = await self._page.query_selector_all(instruction.selector)
        if not elements:
            if instruction.multiple:
                return []
            return None

        extracted_values = []
        for el in elements:
            value = None
            if instruction.attribute:
                value = await el.get_attribute(instruction.attribute)
            else:
                value = await el.text_content()
            if value is not None:
                extracted_values.append(value.strip())
        
        result_key = instruction.selector # Use selector as key for extracted data
        if instruction.multiple:
            self._extracted_data[result_key] = extracted_values
            return extracted_values
        else:
            final_value = extracted_values[0] if extracted_values else None
            self._extracted_data[result_key] = final_value
            return final_value

    async def _handle_navigate(self, instruction: NavigateInstruction):
        if not self._page: raise BrowserAgentError("Page not available")
        try:
            await self._page.goto(
                instruction.url, 
                wait_until=instruction.wait_until.value if instruction.wait_until else "load", 
                timeout=instruction.timeout or self.default_timeout
            )
        except Exception as e:
            raise InstructionExecutionError(instruction.dict(), f"Navigation to {instruction.url} failed: {e}")

    async def _handle_evaluate(self, instruction: EvaluateInstruction) -> Any:
        if not self._page: raise BrowserAgentError("Page not available")
        try:
            result = await self._page.evaluate(instruction.script)
            if instruction.return_by_value:
                # Store result with a generic key or one derived from script if possible
                eval_key = f"eval_result_{len(self._extracted_data)}"
                self._extracted_data[eval_key] = result
            return result
        except Exception as e:
            raise InstructionExecutionError(instruction.dict(), f"JavaScript evaluation failed: {e}")

    async def _handle_upload(self, instruction: UploadInstruction):
        if not self._page: raise BrowserAgentError("Page not available")
        element = await self._get_element(instruction.selector, instruction.timeout)
        # Ensure files exist before attempting upload
        for file_path in instruction.files:
            if not Path(file_path).is_file():
                raise InstructionExecutionError(instruction.dict(), f"File not found for upload: {file_path}")
        await element.set_input_files(instruction.files)
        if not instruction.no_wait_after:
             await self._page.wait_for_load_state("networkidle", timeout=instruction.timeout or self.default_timeout)

    async def _handle_download(self, instruction: DownloadInstruction):
        if not self._page or not self._context: raise BrowserAgentError("Page or context not available")
        
        # Ensure downloads directory exists (should be handled in __init__)
        # self.downloads_path
        target_path = Path(self.downloads_path) / instruction.save_as

        async with self._page.expect_download(timeout=instruction.timeout or self.default_timeout) as download_info:
            # This assumes the download is triggered by clicking an element.
            # If download is triggered differently, this part needs adjustment.
            if instruction.selector: # If a selector is provided, click it to trigger download
                 element_to_click = await self._get_element(instruction.selector, instruction.timeout)
                 await element_to_click.click()
            # If no selector, the download must be triggered by a previous action (e.g. navigation)
        
        download = await download_info.value
        await download.save_as(target_path)
        logger.info(f"File downloaded to: {target_path}")
        # Store path to downloaded file
        self._extracted_data[f"downloaded_file_{instruction.save_as.replace('.', '_')}"] = str(target_path)

    # Public API methods matching the prompt's Technical Specifications
    async def take_screenshot(self, full_page=False, filename: Optional[str] = None) -> str:
        """Capture screenshot and return base64 or file path."""
        instruction = ScreenshotInstruction(
            type=ActionType.SCREENSHOT, 
            full_page=full_page, 
            filename=filename,
            save_to_disk=bool(filename), # Save if filename is provided
            return_as_base64=not bool(filename) # Return base64 if no filename
        )
        return await self._handle_screenshot(instruction) or ""

    async def extract_page_data(self, selectors: List[str]) -> Dict[str, Any]:
        """Extract data from specified page elements."""
        extracted_results = {}
        for selector in selectors:
            instruction = ExtractInstruction(type=ActionType.EXTRACT, selector=selector, multiple=False) # Assuming single extract per selector
            try:
                value = await self._handle_extract(instruction)
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
    def _init_reasoning_engine(self):
        """Initialize reasoning engine if enabled"""
        if not hasattr(self, '_reasoning_engine') and reasoning_config.enabled:
            try:
                self._reasoning_engine = WebAutomationReasoner(
                    browser_agent=self,
                    memory_manager=self.memory_manager
                )
                logger.info("Reasoning engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize reasoning engine: {e}")
                self._reasoning_engine = None
        elif not reasoning_config.enabled:
            self._reasoning_engine = None

    async def execute_instructions_with_reasoning(self, instructions_json: Union[Dict, InstructionSet]) -> Dict:
        """Enhanced instruction execution with optional CoT reasoning"""
        self._init_reasoning_engine() # Ensure it's initialized if config changed
        
        # Convert to InstructionSet if needed
        if isinstance(instructions_json, dict):
            try:
                instruction_set = InstructionSet(**instructions_json)
            except ValidationError as e:
                self._errors.append({"type": "ValidationError", "message": str(e)})
                return self._prepare_result(success=False)
        else:
            instruction_set = instructions_json
        
        # Apply reasoning if enabled and available
        reasoning_result = {}
        if self._reasoning_engine and reasoning_config.enabled:
            logger.info("Applying CoT reasoning to instruction set")
            reasoning_result = await self._reasoning_engine.reason_about_instruction_set(instruction_set)
            
            if reasoning_result.get("reasoning_applied") and reasoning_result.get("result"):
                logger.info(f"Reasoning completed in {reasoning_result.get('execution_time', 0):.2f}s")
                # Potentially modify instruction_set based on reasoning_result['result']
                # For now, we'll just log and proceed with original/modified instructions
                # This part would need careful design on how reasoning output translates to executable actions
            elif not reasoning_result.get("reasoning_applied"):
                 logger.warning(f"Reasoning was not applied or failed: {reasoning_result.get('error', 'unknown')}")
        
        # Execute instructions using existing method
        # This might use the original instruction_set or one modified by reasoning
        execution_result = await self.execute_instructions(instruction_set)
        
        # Add reasoning info to result
        if reasoning_result:
            execution_result["reasoning"] = reasoning_result
            if self._reasoning_engine:
                execution_result["reasoning_stats"] = self._reasoning_engine.get_reasoning_stats()
        
        return execution_result

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get reasoning engine statistics"""
        if hasattr(self, '_reasoning_engine') and self._reasoning_engine:
            return self._reasoning_engine.get_reasoning_stats()
        return {"reasoning_enabled": reasoning_config.enabled, "status": "not_initialized_or_disabled"}


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

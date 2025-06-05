import asyncio
import sys
import base64
import json
import logging
import os
import random
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
from web_automation.captcha.captcha_handler import SimpleCaptchaHandler
from pydantic import ValidationError

from ..config import settings
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

class WebBrowserAgent:
    """
    A browser automation agent that executes JSON-based instructions using Playwright.
    """
    
    def __init__(
        self, 
        browser_type: Optional[str] = None, 
        headless: Optional[bool] = None, 
        stealth: bool = True,
        default_timeout: Optional[int] = None, 
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
        proxy: Optional[Dict[str, str]] = None,
        downloads_path: Optional[str] = None,
        randomize_fingerprint: Optional[bool] = None, 
        custom_fingerprint_script: Optional[str] = None,
        identity_id: Optional[str] = None
    ):
        self.identity_id = identity_id or str(uuid.uuid4())
        self.browser_type = (browser_type or settings.browser_config.DEFAULT_BROWSER_TYPE).lower()
        self.headless = headless if headless is not None else settings.browser_config.DEFAULT_HEADLESS_MODE
        self.stealth = stealth
        self.default_timeout = default_timeout or settings.general_config.DEFAULT_TIMEOUT
        self.viewport = viewport or {"width": settings.browser_config.DEFAULT_VIEWPORT_WIDTH, "height": settings.browser_config.DEFAULT_VIEWPORT_HEIGHT}
        self.user_agent = user_agent # Will be randomized in initialize() if None and randomize_fingerprint is True
        
        if proxy is not None:
            self.proxy = proxy
        elif settings.proxy_config.USE_PROXY and settings.proxy_config.PROXY_SERVER:
            self.proxy = {
                "server": settings.proxy_config.PROXY_SERVER,
            }
            if settings.proxy_config.PROXY_USERNAME and settings.proxy_config.PROXY_PASSWORD:
                self.proxy["username"] = settings.proxy_config.PROXY_USERNAME
                self.proxy["password"] = settings.proxy_config.PROXY_PASSWORD
        else:
            self.proxy = None
            
        self.downloads_path = downloads_path or settings.general_config.DOWNLOADS_DIR
        # Ensure downloads_path exists (already done by general_config instantiation if using default)
        # but good to ensure if a custom path is provided.
        os.makedirs(self.downloads_path, exist_ok=True)

        # Fingerprint options
        self.randomize_fingerprint = randomize_fingerprint if randomize_fingerprint is not None else self.stealth
        self.custom_fingerprint_script = custom_fingerprint_script
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._fingerprint_profile: Dict = {}
    
        self._reset_execution_state()
        self._fingerprint_profile = self._load_or_create_profile()

    def _reset_execution_state(self):
        self._screenshots = []
        self._actions_completed = 0
        self._extracted_data = {}
        self._errors = []
        self._captchas_solved = 0
        self._execution_start_time = None

    def _load_or_create_profile(self) -> Dict:
        profiles_dir_name = "profiles"
        # Assuming this file (browser_agent.py) is in web_automation/core/
        # So, ../../profiles would be at the same level as the web_automation directory.
        # Let's place it inside web_automation/profiles for better module organization.
        base_path = Path(__file__).resolve().parent.parent # web_automation directory
        profiles_dir = base_path / profiles_dir_name
        
        os.makedirs(profiles_dir, exist_ok=True)
        
        profile_file_path = profiles_dir / f"{self.identity_id}.json"
        
        if profile_file_path.exists():
            logger.info(f"PROFILE: Loading existing profile for identity {self.identity_id} from {profile_file_path}")
            try:
                with open(profile_file_path, 'r') as f:
                    profile = json.load(f)
                # Basic validation or schema check could be added here
                if not isinstance(profile, dict) or not profile.get("userAgent"):
                    logger.warning(f"PROFILE: Invalid or empty profile loaded for {self.identity_id}. Regenerating.")
                    raise FileNotFoundError("Invalid profile format") # Trigger regeneration
                return profile
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"PROFILE: Failed to load or parse profile for {self.identity_id} ({e}). Regenerating.")
        
        logger.info(f"PROFILE: No existing profile found for identity {self.identity_id}. Creating new one.")
        # Note: create_consistent_fingerprint might take arguments in the future
        # For now, assuming it generates a full, consistent fingerprint dict
        new_profile = create_consistent_fingerprint()
        
        try:
            with open(profile_file_path, 'w') as f:
                json.dump(new_profile, f, indent=4)
            logger.info(f"PROFILE: Saved new profile for identity {self.identity_id} to {profile_file_path}")
        except IOError as e:
            logger.error(f"PROFILE: Failed to save new profile for {self.identity_id} to {profile_file_path}: {e}")
            # If saving fails, we still return the generated profile for the current session
            # but it won't be persisted.

        return new_profile

    async def _human_like_delay(self, min_ms: Optional[int] = None, max_ms: Optional[int] = None, base_delay_type: str = "before"):
        """Introduces a random delay. Uses config defaults if min/max not provided."""
        if base_delay_type == "before":
            min_d = min_ms if min_ms is not None else anti_detection_config.MIN_DELAY_BEFORE_ACTION
            max_d = max_ms if max_ms is not None else anti_detection_config.MAX_DELAY_BEFORE_ACTION
        elif base_delay_type == "after":
            min_d = min_ms if min_ms is not None else anti_detection_config.MIN_DELAY_AFTER_ACTION
            max_d = max_ms if max_ms is not None else anti_detection_config.MAX_DELAY_AFTER_ACTION
        elif base_delay_type == "typing":
            min_d = min_ms if min_ms is not None else anti_detection_config.MIN_TYPING_DELAY_PER_CHAR
            max_d = max_ms if max_ms is not None else anti_detection_config.MAX_TYPING_DELAY_PER_CHAR
        else: # Default to 'before' action style delays
            min_d = min_ms if min_ms is not None else anti_detection_config.MIN_DELAY_BEFORE_ACTION
            max_d = max_ms if max_ms is not None else anti_detection_config.MAX_DELAY_BEFORE_ACTION

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
            print("STEALTH: Context or Page not available, skipping playwright-stealth application.")
            return

        try:
            logger.info("STEALTH: Attempting to apply playwright-stealth protection...")
            print("STEALTH: Attempting to apply playwright-stealth protection...")
            await stealth_async(self._page)
            logger.info("STEALTH: Successfully applied playwright-stealth protection.")
            print("STEALTH: Successfully applied playwright-stealth protection.")
        except Exception as e:
            logger.error(f"STEALTH: Error applying playwright-stealth: {e}", exc_info=True)
            print(f"STEALTH: Error applying playwright-stealth: {e}")
            # Decide if this should re-raise or if the agent can continue without stealth
            # For now, logging the error and continuing.

        # The base_stealth_script for navigator.webdriver is handled by playwright-stealth
        # If specific other init scripts are still needed, they could be added here.
        # For now, assuming playwright-stealth covers the necessary basics.

        logger.info("STEALTH: _enable_stealth completed using playwright-stealth.")
        print("STEALTH: _enable_stealth completed using playwright-stealth.")

    async def execute_instructions(self, instructions_json: Union[Dict, InstructionSet]) -> Dict:
        self._reset_execution_state()
        self._execution_start_time = time.time()

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

        if instruction_set.url:
            try:
                await self._handle_navigate(NavigateInstruction(type=ActionType.NAVIGATE, url=instruction_set.url))
            except Exception as e:
                self._errors.append({"type": "NavigationError", "message": str(e), "url": instruction_set.url})
                return self._prepare_result(success=False)

        for instruction_data in instruction_set.instructions:
            try:
                action_type = ActionType(instruction_data.type)
                handler = self._get_action_handler(action_type)

                if not handler:
                    logger.error(f"Unsupported instruction type: {action_type.value}")
                    self._errors.append({
                        "type": "UnsupportedInstructionError",
                        "instruction_type": action_type.value,
                        "message": f"No handler found for action type '{action_type.value}'"
                    })
                    continue # Skip to the next instruction

                # Apply pre-action delay for most actions
                if action_type not in [ActionType.WAIT, ActionType.EVALUATE]: # Don't delay before wait or eval
                    await self._human_like_delay(base_delay_type="before")

                await handler(instruction_data)
                self._actions_completed += 1
                await self._human_like_delay(base_delay_type="after")
            except InstructionExecutionError as e:
                self._errors.append({
                    "type": "InstructionExecutionError", 
                    "instruction_type": e.instruction.get('type'),
                    "selector": e.instruction.get('selector'),
                    "message": e.message
                })
                # Optionally, decide whether to stop or continue on error
                # For now, let's stop on first error
                return self._prepare_result(success=False)
            except Exception as e:
                self._errors.append({
                    "type": "GenericError",
                    "instruction_type": instruction_data.type.value if hasattr(instruction_data, 'type') else 'unknown',
                    "message": str(e)
                })
                return self._prepare_result(success=False)
        
        return self._prepare_result(success=True)

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

    async def solve_captcha(self, captcha_type: str) -> bool:
        """Handle various CAPTCHA types (Placeholder)."""
        # This is a placeholder for CAPTCHA solving integration.
        # Actual implementation would involve calling a CAPTCHA service.
        logger.warning(f"CAPTCHA solving for type '{captcha_type}' is not yet implemented.")
        # Simulate a failed attempt for now
        self._errors.append({"type": "CaptchaError", "message": f"CAPTCHA type '{captcha_type}' not solvable."})
        return False


# Apply CAPTCHA integration methods to the WebBrowserAgent class
WebBrowserAgent = CaptchaIntegration.add_captcha_methods(WebBrowserAgent)

async def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    print("Initializing WebBrowserAgent for CAPTCHA test...")
    agent = WebBrowserAgent(headless=False, stealth=True) # Ensure headless is False for visibility
    captcha_handler = SimpleCaptchaHandler() # Initialize the CAPTCHA handler

    async with agent:
        print("WebBrowserAgent entered context. Navigating to reCAPTCHA test page...")
        
        # Navigate to a reCAPTCHA v2 test page
        # You might need to find a reliable public test page, or set up your own.
        # For demonstration, we'll use a common one, but it might change or have rate limits.
        recaptcha_test_url = "https://www.google.com/recaptcha/api2/demo"
        await agent._page.goto(recaptcha_test_url, wait_until="domcontentloaded")
        logger.info(f"Navigated to: {recaptcha_test_url}")

        print("Attempting to handle CAPTCHA...")
        # Use the captcha_handler to detect and solve CAPTCHAs on the current page
        captcha_solved = await captcha_handler.handle_page_captchas(agent._page)
        
        if captcha_solved:
            logger.info("CAPTCHA successfully handled!")
            print("CAPTCHA successfully handled!")
        else:
            logger.warning("Failed to handle CAPTCHA.")
            print("Failed to handle CAPTCHA.")

        print("CAPTCHA handling complete. Pausing for human inspection (30 seconds)...")
        print("You can inspect the browser window to see the result.")
        await asyncio.sleep(30) # Pause for human inspection

        print("WebBrowserAgent exited context. Test finished.")
        print(f"CAPTCHA Handler Stats: {captcha_handler.get_stats()}")

    print("WebBrowserAgent exited context. Test finished.")

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

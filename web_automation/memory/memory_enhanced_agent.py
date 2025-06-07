import logging
import re
import asyncio
import random
import base64 # Added for screenshot encoding
from typing import Optional, Union

from ..models.instructions import NavigateInstruction, ClickInstruction, TypeInstruction

from ..core.browser_agent import PlaywrightBrowserAgent, BrowserAgentError
from web_automation.memory.memory_manager import Mem0BrowserAdapter
from web_automation.models.instructions import ActionType, NavigateInstruction
from ..config.config_models import Mem0AdapterConfig
from ..vision.visual_memory_system import VisualMemorySystem
from web_automation.vision.image_analyzer import ImageAnalyzer

logger = logging.getLogger(__name__)

class PersistentMemoryBrowserAgent(PlaywrightBrowserAgent):
    """
    PlaywrightBrowserAgent with enhanced memory and explicit visual analysis capabilities.
    Visual memory (auto-capture) is disabled by default and only enabled if both enabled and auto_capture are True.
    Explicit image analysis is always available if the visual system is configured.
    """

    def __init__(self, dependencies, **kwargs):
        super().__init__(dependencies=dependencies, **kwargs)
        logger.info("PersistentMemoryBrowserAgent attempting initialization.")

        self.memory_manager = getattr(dependencies, 'memory_manager', None)
        self.memory_enabled = self.memory_manager is not None
        if self.memory_enabled:
            logger.info(f"Memory manager configured: {self.memory_manager.__class__.__name__}")
        else:
            logger.warning("No memory manager provided. Memory features disabled.")

        self.visual_system: Optional[VisualMemorySystem] = None
        self.visual_system_enabled: bool = False
        self.visual_auto_capture: bool = False
        ollama_client = getattr(dependencies, 'ollama_client', None)
        visual_llm_model_name = getattr(dependencies, 'visual_llm_model_name', None)

        # VisualSystemConfig is always passed as dict or model; check for enabled/auto_capture
        visual_config = kwargs.get('visual_config_input') or getattr(dependencies, 'visual_config', None)
        print(f"DEBUG: visual_config received in PersistentMemoryBrowserAgent: {visual_config}")
        # Defensive: ensure dict
        if hasattr(visual_config, 'model_dump'):
            visual_config = visual_config.model_dump()
        print(f"DEBUG: visual_config after model_dump: {visual_config}")
        if not visual_config:
            visual_config = {'enabled': False, 'auto_capture': False}
        enabled = bool(visual_config.get('enabled', False))
        auto_capture = bool(visual_config.get('auto_capture', False))
        print(f"DEBUG: enabled: {enabled}, auto_capture: {auto_capture}")
        self.visual_auto_capture = enabled and auto_capture

        if enabled and ollama_client and visual_llm_model_name and self.memory_manager:
            try:
                self.visual_system = VisualMemorySystem(
                    llm_client=ollama_client,
                    memory_manager=self.memory_manager,
                    llm_model_name=visual_llm_model_name
                )
                self.visual_system_enabled = True
                logger.info(f"VisualMemorySystem initialized with model '{visual_llm_model_name}'. Visual fallback/analysis enabled.")
            except Exception as e:
                logger.error(f"Failed to initialize VisualMemorySystem: {e}. Visual features disabled.")
                self.visual_system = None
                self.visual_system_enabled = False
        else:
            logger.info("VisualMemorySystem not initialized (disabled or missing dependencies). Visual fallback/analysis disabled.")

        # Initialize session statistics
        self.interactions_stored_session = 0

    async def _capture_visuals_if_enabled(self, action_type: str, target_selector: Optional[str] = None):
        if not self.visual_system_enabled or not self.visual_system or not self.visual_auto_capture:
            return
        try:
            if self._page and not self._page.is_closed():
                current_url = self._page.url
                logger.debug(f"Capturing visual context for action: {action_type} at URL: {current_url}")
                await self.visual_system.capture_visual_context(
                    page=self._page,
                    user_id=self.identity_id,
                    action_type=action_type,
                    target_element_selector=target_selector,
                    current_url=current_url
                )
            else:
                logger.warning("Page not available or closed, skipping visual context capture.")
        except Exception as e:
            logger.error(f"Error during visual context capture for {action_type}: {e}", exc_info=True)

    async def navigate(self, url: str, **kwargs) -> None:
        nav_instruction = NavigateInstruction(type=ActionType.NAVIGATE, url=url, **kwargs)
        await self._execute_instruction_with_memory(nav_instruction, self.identity_id)
        await self._capture_visuals_if_enabled(action_type="navigate_complete", target_selector=url)

    async def click(self, selector: str, **kwargs) -> None:
        click_instruction = ClickInstruction(type=ActionType.CLICK, selector=selector, **kwargs)
        await self._execute_instruction_with_memory(click_instruction, self.identity_id)
        await self._capture_visuals_if_enabled(action_type="click_complete", target_selector=selector)


    async def fill(self, selector: str, value: str, **kwargs) -> None:
        type_instruction = TypeInstruction(selector=selector, text=value, **kwargs)
        await super()._handle_type(type_instruction)
        await self._capture_visuals_if_enabled(action_type="fill_complete", target_selector=selector)

    async def smart_selector_click(self, target_description: str, fallback_selector: Optional[str] = None, timeout: Optional[int] = None) -> bool:
        logger.info(f"Attempting smart_selector_click for: '{target_description}' with fallback: '{fallback_selector}'")
        
        # Attempt standard click methods first
        success = False
        reason_or_selector_used = ""
        page_url = self._page.url if self._page and not self._page.is_closed() else "unknown_url"

        primary_selector_to_try = target_description # Assuming target_description can be a selector or smart string
        
        try:
            logger.info(f"Attempting smart_selector_click with primary selector/description: '{primary_selector_to_try}' at URL: {page_url}")
            await self.click(primary_selector_to_try, timeout=timeout) # Calls PersistentMemoryBrowserAgent.click
            success = True
            reason_or_selector_used = primary_selector_to_try
            logger.info(f"smart_selector_click: Primary click attempt SUCCEEDED with '{primary_selector_to_try}'.")
        except Exception as e_primary:
            logger.warning(f"smart_selector_click: Primary click attempt with '{primary_selector_to_try}' FAILED: {e_primary}")
            if fallback_selector:
                logger.info(f"Attempting smart_selector_click with fallback selector: '{fallback_selector}' at URL: {page_url}")
                try:
                    await self.click(fallback_selector, timeout=timeout)
                    success = True
                    reason_or_selector_used = fallback_selector
                    logger.info(f"smart_selector_click: Fallback click attempt SUCCEEDED with '{fallback_selector}'.")
                except Exception as e_fallback:
                    logger.warning(f"smart_selector_click: Fallback click attempt with '{fallback_selector}' also FAILED: {e_fallback}")
                    reason_or_selector_used = f"Primary click failed: {type(e_primary).__name__} - {str(e_primary)}. Fallback click failed: {type(e_fallback).__name__} - {str(e_fallback)}."
                    # success remains False
            else:
                reason_or_selector_used = f"Primary click failed: {type(e_primary).__name__} - {str(e_primary)}. No fallback selector provided."
                # success remains False


        if success:
            logger.info(f"smart_selector_click SUCCEEDED for '{target_description}' using: {reason_or_selector_used}")
            if self.memory_manager:
                success_metadata = {
                    "action_type": "smart_selector_click",
                    "status": "success",
                    "url": page_url,
                    "original_target_description": target_description # Keep original if needed
                }
                self.memory_manager.store_automation_pattern(
                    user_id=self.identity_id,
                    description=target_description, # Mapped from target_description
                    selector=reason_or_selector_used,    # Mapped from selector_used
                    success=True,
                    fallback_selector=fallback_selector, # Passed as original_fallback_selector
                    metadata=success_metadata
                )
                self.interactions_stored_session += 1
            await self._capture_visuals_if_enabled(action_type="smart_selector_click_success", target_selector=reason_or_selector_used)
            return True
        else:
            # Standard selectors failed, log this failure
            logger.warning(f"smart_selector_click FAILED for '{target_description}' using standard selectors. Reason: {reason_or_selector_used}")
            if self.memory_manager:
                failure_metadata = {
                    "action_type": "smart_selector_click",
                    "status": "failure_selectors",
                    "failure_reason": reason_or_selector_used,
                    "url": page_url,
                    "original_target_description": target_description # Keep original if needed
                }
                self.memory_manager.store_automation_pattern(
                    user_id=self.identity_id,
                    description=target_description, # Mapped from target_description
                    selector=fallback_selector or "N/A", # Mapped from selector_used (or N/A)
                    success=False,
                    fallback_selector=fallback_selector, # Passed as original_fallback_selector
                    metadata=failure_metadata
                )
                self.interactions_stored_session += 1
            
            # Attempt visual fallback if enabled and page is available
            if self.visual_system_enabled and self.visual_system and self._page and not self._page.is_closed():
                logger.info(f"Attempting visual fallback for '{target_description}' at URL {page_url}.")
                current_screenshot_base64 = None
                try:
                    screenshot_bytes = await self._page.screenshot(type='png', full_page=True)
                    current_screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to take screenshot for visual fallback: {e}")
                    # VMS will handle None screenshot if it occurs, storing a pattern indicating this issue.

                try:
                    visual_fallback_succeeded = await self.visual_system.enable_visual_fallback(
                        page=self._page,
                        user_id=self.identity_id,
                        failed_action_description=target_description,
                        failed_selector=fallback_selector,
                        current_screenshot_base64=current_screenshot_base64
                    )
                    
                    if visual_fallback_succeeded:
                        logger.info(f"Visual fallback SUCCEEDED for '{target_description}'.")
                        # VisualMemorySystem handles storing its own success pattern, including visuals.
                        # No need to call _capture_visuals_if_enabled here as VMS already has the screenshot.
                        return True
                    else:
                        logger.warning(f"Visual fallback FAILED for '{target_description}'. Outcome logged by VisualMemorySystem.")
                        return False # VMS logged the specific reason for visual fallback failure
                except Exception as e:
                    logger.error(f"Exception during visual_system.enable_visual_fallback call for '{target_description}': {e}", exc_info=True)
                    # If enable_visual_fallback itself raises an unhandled exception, it's a system error.
                    # VMS should ideally catch internal errors, but if not, this path is hit.
                    # Storing a generic visual system error might be an option here if VMS doesn't.
                    return False
            elif not self.visual_system_enabled:
                logger.info("Visual system not enabled. No visual fallback attempted.")
                return False
            else: # Page closed or not available
                logger.warning(f"Page not available for visual fallback for '{target_description}'.")
                return False

    def get_memory_stats(self) -> dict:
        """Return basic statistics about the current memory session."""
        if not self.memory_manager or not hasattr(self.memory_manager, 'memory') or not self.memory_manager.memory:
            return {"memory_enabled": False}
        
        # Assuming get_session_context might not exist or be relevant for all Mem0 versions/adapters.
        # A more generic check or relying on specific Mem0 features if available.
        # For now, let's keep it simple if direct session context isn't universally applicable.
        # interactions_count = len(self.memory_manager.memory.get_all(user_id=self.identity_id) or [])
        # Awaiting a more robust way to get session-specific stats from Mem0 if possible.
        # For now, just indicate memory is enabled.
        return {
            "memory_enabled": True,
            "interactions_stored_session": self.interactions_stored_session
        }

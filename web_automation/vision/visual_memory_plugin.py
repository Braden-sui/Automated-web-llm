from typing import Any, Optional, Dict
from web_automation.core.plugin_base import AgentPlugin
from web_automation.vision.visual_memory_system import VisualMemorySystem
import logging
import ollama

logger = logging.getLogger(__name__)

class VisualMemoryPlugin(AgentPlugin):
    """
    Plugin for handling visual memory capabilities in the web automation framework.
    """
    def __init__(self, visual_system: Optional[VisualMemorySystem] = None):
        self.visual_system = visual_system
        self.agent = None

    def initialize(self, agent: Any) -> None:
        """
        Initialize the Visual Memory plugin with the agent instance.

        Args:
            agent: The agent instance to which this plugin is attached.
        """
        self.agent = agent
        if self.visual_system is None:
            from web_automation.config.config_models import VisualSystemConfig
            from web_automation.config.settings import visual_system_config as config
            if not isinstance(config, VisualSystemConfig):
                config = VisualSystemConfig(**config) if isinstance(config, dict) else VisualSystemConfig()
            if config.enabled:
                ollama_client = ollama.AsyncClient(host=config.ollama_base_url or "http://localhost:11434")
                self.visual_system = VisualMemorySystem(ollama_client, None, config.model_name)
                logger.info(f"Visual Memory System initialized with model {config.model_name}")
            else:
                logger.info("Visual Memory System is disabled in configuration")
        else:
            logger.info("Visual Memory Plugin initialized with provided system")

    def get_name(self) -> str:
        """
        Return the name of the plugin.

        Returns:
            str: The name of the plugin.
        """
        return "visual_memory"

    def is_enabled(self) -> bool:
        """
        Check if the visual memory system is enabled.

        Returns:
            bool: True if enabled, False otherwise.
        """
        return self.visual_system is not None and hasattr(self.visual_system, 'visual_system_enabled') and self.visual_system.visual_system_enabled

    async def capture_visual_context(self, screenshot_base64: str, url: str) -> Dict:
        """
        Capture visual context from a screenshot.

        Args:
            screenshot_base64: Base64 encoded screenshot.
            url: URL of the page where the screenshot was taken.

        Returns:
            Dict: Visual context data.
        """
        if not self.is_enabled():
            logger.warning("Visual Memory System is not enabled, skipping capture")
            return {}
        try:
            return await self.visual_system.capture_visual_context(screenshot_base64, url)
        except Exception as e:
            logger.error(f"Error capturing visual context: {e}", exc_info=True)
            return {}

    async def match_visual_pattern(self, screenshot_base64: str, url: str, action_type: str, target_element_selector: Optional[str] = None) -> Optional[Dict]:
        """
        Match current visual state against stored patterns.

        Args:
            screenshot_base64: Base64 encoded screenshot of the current page.
            url: URL of the current page.
            action_type: Type of action to match (e.g., 'click', 'type').
            target_element_selector: CSS selector of the target element, if any.

        Returns:
            Optional[Dict]: Matched pattern data if found, None otherwise.
        """
        if not self.is_enabled():
            logger.warning("Visual Memory System is not enabled, skipping pattern match")
            return None
        try:
            return await self.visual_system.match_visual_pattern(screenshot_base64, url, action_type, target_element_selector)
        except Exception as e:
            logger.error(f"Error matching visual pattern: {e}", exc_info=True)
            return None

    async def store_visual_pattern(self, screenshot_base64: str, url: str, action_type: str, target_element_selector: str, success: bool, visual_context_metadata: Dict) -> None:
        """
        Store a visual pattern for future reference.

        Args:
            screenshot_base64: Base64 encoded screenshot.
            url: URL of the page.
            action_type: Type of action performed.
            target_element_selector: CSS selector of the target element.
            success: Whether the action was successful.
            visual_context_metadata: Additional metadata about the visual context.
        """
        if not self.is_enabled():
            logger.warning("Visual Memory System is not enabled, skipping store")
            return
        try:
            await self.visual_system.store_visual_pattern(screenshot_base64, url, action_type, target_element_selector, success, visual_context_metadata)
        except Exception as e:
            logger.error(f"Error storing visual pattern: {e}", exc_info=True)

    async def enable_visual_fallback(self, enable: bool = True) -> None:
        """
        Enable or disable visual fallback for automation.

        Args:
            enable: Boolean to enable or disable fallback.
        """
        if not self.is_enabled():
            logger.warning("Visual Memory System is not enabled, cannot change fallback state")
            return
        try:
            await self.visual_system.enable_visual_fallback(enable)
        except Exception as e:
            logger.error(f"Error enabling visual fallback: {e}", exc_info=True)

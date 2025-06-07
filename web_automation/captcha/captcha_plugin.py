from typing import Any, Optional
from web_automation.core.plugin_base import AgentPlugin
from web_automation.captcha.vision_handler import VisionCaptchaHandler
import logging

logger = logging.getLogger(__name__)

class CaptchaPlugin(AgentPlugin):
    """
    Plugin for handling CAPTCHA challenges in the web automation framework.
    """
    def __init__(self, handler: Optional[VisionCaptchaHandler] = None):
        self.handler = handler
        self.agent = None

    def initialize(self, agent: Any) -> None:
        """
        Initialize the CAPTCHA plugin with the agent instance.

        Args:
            agent: The agent instance to which this plugin is attached.
        """
        self.agent = agent
        if self.handler is None:
            from web_automation.config.settings import captcha_config
            self.handler = VisionCaptchaHandler(captcha_config)
        logger.info("CAPTCHA Plugin initialized for agent")

    def get_name(self) -> str:
        """
        Return the name of the plugin.

        Returns:
            str: The name of the plugin.
        """
        return "captcha"

    async def solve(self) -> bool:
        """
        Attempt to solve a CAPTCHA challenge on the current page.

        Returns:
            bool: True if CAPTCHA was solved successfully, False otherwise.
        """
        if not self.agent or not self.agent._page:
            logger.error("Cannot solve CAPTCHA: Agent or page not initialized")
            return False
        try:
            logger.info("Attempting to solve CAPTCHA")
            return await self.handler.solve_captcha(self.agent._page)
        except Exception as e:
            logger.error(f"Error solving CAPTCHA: {e}", exc_info=True)
            return False

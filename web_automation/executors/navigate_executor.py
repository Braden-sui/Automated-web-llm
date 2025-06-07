from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class NavigateExecutor(BaseExecutor):
    """
    Executor for handling navigation actions to different URLs.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a navigation action to the specified URL.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing navigation action details.

        Returns:
            Any: Result of the navigation action.
        """
        try:
            url = instruction.url if hasattr(instruction, 'url') else instruction.get('url', '')
            logger.info(f"Navigating to URL: {url}")
            await page.goto(url)
            logger.info(f"Successfully navigated to: {url}")
            return True
        except Exception as e:
            logger.error(f"Error navigating to URL {url}: {e}", exc_info=True)
            raise

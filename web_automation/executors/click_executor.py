from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class ClickExecutor(BaseExecutor):
    """
    Executor for handling click actions on web elements.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a click action on the specified element.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing click action details.

        Returns:
            Any: Result of the click action.
        """
        try:
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', '')
            logger.info(f"Clicking element with selector: {selector}")
            await page.click(selector)
            logger.info(f"Successfully clicked element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Error clicking element with selector {selector}: {e}", exc_info=True)
            raise

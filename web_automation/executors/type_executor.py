from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class TypeExecutor(BaseExecutor):
    """
    Executor for handling typing actions into web elements.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a typing action into the specified element.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing typing action details.

        Returns:
            Any: Result of the typing action.
        """
        try:
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', '')
            text = instruction.text if hasattr(instruction, 'text') else instruction.get('text', '')
            logger.info(f"Typing '{text}' into element with selector: {selector}")
            await page.fill(selector, text)
            logger.info(f"Successfully typed into element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Error typing into element with selector {selector}: {e}", exc_info=True)
            raise

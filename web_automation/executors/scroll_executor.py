from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging
import asyncio

import random

logger = logging.getLogger(__name__)

class ScrollExecutor(BaseExecutor):
    """
    Executor for handling scroll actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a scroll action on the page.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing scroll action details.

        Returns:
            Any: Result of the scroll action.
        """
        try:
            scroll_into_view = instruction.scroll_into_view if hasattr(instruction, 'scroll_into_view') else instruction.get('scroll_into_view', False)
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', None)
            x = instruction.x if hasattr(instruction, 'x') else instruction.get('x', None)
            y = instruction.y if hasattr(instruction, 'y') else instruction.get('y', None)
            behavior = instruction.behavior if hasattr(instruction, 'behavior') else instruction.get('behavior', 'smooth')
            timeout = instruction.timeout if hasattr(instruction, 'timeout') else instruction.get('timeout', 5000)

            logger.info(f"Scrolling with behavior: {behavior}")

            if scroll_into_view and selector:
                element = await page.wait_for_selector(selector, state="visible", timeout=timeout)
                if element:
                    await element.scroll_into_view_if_needed(timeout=timeout)
                else:
                    logger.warning(f"Element with selector {selector} not found for scrolling")
                    return False
            elif x is not None or y is not None:
                script = f"window.scrollBy({{ left: {x or 0}, top: {y or 0}, behavior: '{behavior}' }})"
                await page.evaluate(script)

            await asyncio.sleep(random.uniform(0.2, 0.5))  # Wait for scroll to take effect
            logger.info(f"Scroll action completed")
            return True
        except Exception as e:
            logger.error(f"Error during scroll action: {e}", exc_info=True)
            raise

from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class WaitExecutor(BaseExecutor):
    """
    Executor for handling wait actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a wait action based on the specified condition.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing wait action details.

        Returns:
            Any: Result of the wait action.
        """
        try:
            timeout = instruction.timeout if hasattr(instruction, 'timeout') else instruction.get('timeout', 5000)
            condition = instruction.condition if hasattr(instruction, 'condition') else instruction.get('condition', 'TIMEOUT')
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', None)
            wait_for = instruction.wait_for if hasattr(instruction, 'wait_for') else instruction.get('wait_for', None)

            logger.info(f"Waiting with condition: {condition}, timeout: {timeout}")

            if condition == 'NAVIGATION':
                await page.wait_for_load_state("load", timeout=timeout)
            elif condition == 'NETWORK_IDLE':
                await page.wait_for_load_state("networkidle", timeout=timeout)
            elif condition == 'ELEMENT_VISIBLE' and selector:
                await page.wait_for_selector(selector, state="visible", timeout=timeout)
            elif condition == 'ELEMENT_HIDDEN' and selector:
                await page.wait_for_selector(selector, state="hidden", timeout=timeout)
            elif condition == 'TIMEOUT' and isinstance(wait_for, (int, float)):
                import asyncio
                await asyncio.sleep(wait_for / 1000)
            else:
                logger.warning(f"Unsupported or misconfigured wait condition: {condition}")
                return False

            logger.info(f"Wait condition {condition} satisfied")
            return True
        except Exception as e:
            logger.error(f"Error during wait action: {e}", exc_info=True)
            raise

from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class EvaluateExecutor(BaseExecutor):
    """
    Executor for handling JavaScript evaluation actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a JavaScript evaluation action on the page.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing evaluate action details.

        Returns:
            Any: Result of the JavaScript evaluation.
        """
        try:
            script = instruction.script if hasattr(instruction, 'script') else instruction.get('script', '')
            return_by_value = instruction.return_by_value if hasattr(instruction, 'return_by_value') else instruction.get('return_by_value', False)

            logger.info(f"Evaluating JavaScript script")

            result = await page.evaluate(script)
            if return_by_value and hasattr(instruction, '_extracted_data'):
                eval_key = f"eval_result_{len(instruction._extracted_data)}"
                instruction._extracted_data[eval_key] = result

            logger.info(f"JavaScript evaluation completed")
            return result
        except Exception as e:
            logger.error(f"Error evaluating JavaScript: {e}", exc_info=True)
            raise

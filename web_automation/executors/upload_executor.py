from typing import Any
from playwright.async_api import Page, ElementHandle
from web_automation.executors.base_executor import BaseExecutor
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class UploadExecutor(BaseExecutor):
    """
    Executor for handling file upload actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a file upload action on the specified element.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing upload action details.

        Returns:
            Any: Result of the upload action.
        """
        try:
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', '')
            files = instruction.files if hasattr(instruction, 'files') else instruction.get('files', [])
            timeout = instruction.timeout if hasattr(instruction, 'timeout') else instruction.get('timeout', 5000)
            no_wait_after = instruction.no_wait_after if hasattr(instruction, 'no_wait_after') else instruction.get('no_wait_after', False)

            logger.info(f"Uploading files to element with selector: {selector}")

            element = await page.wait_for_selector(selector, state="visible", timeout=timeout)
            if not element:
                raise ValueError(f"Element not found or not visible: {selector}")

            # Ensure files exist before attempting upload
            for file_path in files:
                if not Path(file_path).is_file():
                    raise ValueError(f"File not found for upload: {file_path}")

            await element.set_input_files(files)
            if not no_wait_after:
                await page.wait_for_load_state("networkidle", timeout=timeout)

            logger.info(f"Successfully uploaded files to element: {selector}")
            return True
        except Exception as e:
            logger.error(f"Error uploading files with selector {selector}: {e}", exc_info=True)
            raise

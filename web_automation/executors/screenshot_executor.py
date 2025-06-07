from typing import Any, Optional
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ScreenshotExecutor(BaseExecutor):
    """
    Executor for handling screenshot actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a screenshot action on the page.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing screenshot action details.

        Returns:
            Any: Result of the screenshot action, typically a dictionary with status and path.
        """
        try:
            full_page = instruction.full_page if hasattr(instruction, 'full_page') else instruction.get('full_page', False)
            filename = instruction.filename if hasattr(instruction, 'filename') else instruction.get('filename', None)

            if not filename:
                filename = f"screenshot_{time.time()}.png"
                logger.debug(f"No filename provided, generated default: {filename}")

            # Ensure the screenshots directory exists
            screenshots_dir = Path("screenshots")  # Adjust path as needed based on config
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            full_path = screenshots_dir / filename
            logger.debug(f"Saving screenshot to: {full_path}")

            await page.screenshot(path=full_path, full_page=full_page)
            logger.info(f"Screenshot taken: {full_path}")

            # Assuming there's a way to store screenshots list in the agent
            if hasattr(instruction, '_screenshots'):
                instruction._screenshots.append(str(full_path))

            return {"status": "success", "path": str(full_path)}
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

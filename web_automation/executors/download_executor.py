from typing import Any
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DownloadExecutor(BaseExecutor):
    """
    Executor for handling file download actions in web automation.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute a file download action triggered by clicking an element or previous action.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing download action details.

        Returns:
            Any: Result of the download action, typically the path to the downloaded file.
        """
        try:
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', None)
            save_as = instruction.save_as if hasattr(instruction, 'save_as') else instruction.get('save_as', 'downloaded_file')
            timeout = instruction.timeout if hasattr(instruction, 'timeout') else instruction.get('timeout', 5000)
            downloads_path = instruction.downloads_path if hasattr(instruction, 'downloads_path') else instruction.get('downloads_path', 'downloads')

            logger.info(f"Downloading file, save as: {save_as}")

            target_path = Path(downloads_path) / save_as

            async with page.expect_download(timeout=timeout) as download_info:
                if selector:  # If a selector is provided, click it to trigger download
                    element = await page.wait_for_selector(selector, state="visible", timeout=timeout)
                    if element:
                        await element.click()
                    else:
                        raise ValueError(f"Element not found for download trigger: {selector}")
                # If no selector, assume download is triggered by a previous action

            download = await download_info.value
            await download.save_as(target_path)
            logger.info(f"File downloaded to: {target_path}")

            # Store path to downloaded file
            if hasattr(instruction, '_extracted_data'):
                instruction._extracted_data[f"downloaded_file_{save_as.replace('.', '_')}"] = str(target_path)

            return str(target_path)
        except Exception as e:
            logger.error(f"Error downloading file: {e}", exc_info=True)
            raise

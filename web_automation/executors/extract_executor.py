from typing import Any, Union, List
from playwright.async_api import Page
from web_automation.executors.base_executor import BaseExecutor
import logging

logger = logging.getLogger(__name__)

class ExtractExecutor(BaseExecutor):
    """
    Executor for handling data extraction actions from web elements.
    """
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute an extract action to retrieve data from web elements.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing extract action details.

        Returns:
            Any: Extracted data, either a single value or a list depending on 'multiple' flag.
        """
        try:
            selector = instruction.selector if hasattr(instruction, 'selector') else instruction.get('selector', '')
            multiple = instruction.multiple if hasattr(instruction, 'multiple') else instruction.get('multiple', False)
            attribute = instruction.attribute if hasattr(instruction, 'attribute') else instruction.get('attribute', None)

            if not selector:
                raise ValueError("Selector is required for extract instruction.")

            logger.info(f"Extracting data from elements with selector: {selector}, multiple: {multiple}")

            elements = await page.query_selector_all(selector)
            if not elements:
                return [] if multiple else None

            extracted_values = []
            for el in elements:
                value = None
                if attribute:
                    value = await el.get_attribute(attribute)
                else:
                    value = await el.text_content()
                if value is not None:
                    extracted_values.append(value.strip())

            result_key = selector  # Use selector as key for extracted data
            if multiple:
                if hasattr(instruction, '_extracted_data'):
                    instruction._extracted_data[result_key] = extracted_values
                return extracted_values
            else:
                final_value = extracted_values[0] if extracted_values else None
                if hasattr(instruction, '_extracted_data'):
                    instruction._extracted_data[result_key] = final_value
                return final_value
        except Exception as e:
            logger.error(f"Error extracting data with selector {selector}: {e}", exc_info=True)
            raise

from abc import ABC, abstractmethod
from typing import Any
from playwright.async_api import Page

class BaseExecutor(ABC):
    """
    Abstract base class for executors that handle specific browser actions.
    """
    @abstractmethod
    async def execute(self, page: Page, instruction: Any) -> Any:
        """
        Execute the specific action on the given page.

        Args:
            page: Playwright Page object to perform the action on.
            instruction: Instruction object containing action details.

        Returns:
            Any: Result of the action execution.
        """
        pass

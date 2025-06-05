import logging
import re
from typing import Optional

from ..core.browser_agent import WebBrowserAgent, BrowserAgentError
from .memory_manager import BrowserMemoryManager
from ..config.config_models import Mem0AIConfig

logger = logging.getLogger(__name__)

class MemoryEnhancedWebBrowserAgent(WebBrowserAgent):
    """WebBrowserAgent with additional helpers that utilise the BrowserMemoryManager."""

    def __init__(self, mem0_config: Optional[Mem0AIConfig] = None, **kwargs):
        # Always enable memory for the enhanced agent
        kwargs.setdefault("memory_enabled", True)
        super().__init__(**kwargs)

        if mem0_config is not None:
            # Recreate memory manager with provided configuration
            try:
                self.memory_manager = BrowserMemoryManager(mem0_config=mem0_config)
                logger.info("Memory manager initialised with custom Mem0 configuration.")
            except Exception as e:
                logger.error(f"Failed to initialise memory manager: {e}")
                self.memory_manager = None

    async def smart_selector_click(self, target_description: str, fallback_selector: str, timeout: int = 5000) -> bool:
        """Attempt click using selectors learned from memory before falling back."""
        selectors_to_try = []

        if self.memory_manager:
            results = self.memory_manager.search_automation_patterns(
                pattern_query=target_description,
                user_id=self.identity_id,
                limit=3,
            )
            for res in results:
                match = re.search(r"selector:\s*(\S+)", res.get("memory", ""))
                if match:
                    selectors_to_try.append(match.group(1))

        selectors_to_try.append(fallback_selector)

        for sel in selectors_to_try:
            try:
                element = await self._get_element(sel, timeout)
                await element.click()
                if self.memory_manager:
                    self.memory_manager.store_automation_pattern(f"selector: {sel}", True, self.identity_id)
                return True
            except Exception as e:
                logger.debug(f"Selector {sel} failed: {e}")
                continue

        if self.memory_manager:
            self.memory_manager.store_automation_pattern(
                f"selector failed: {target_description}", False, self.identity_id
            )
        return False

    def get_memory_stats(self) -> dict:
        """Return basic statistics about the current memory session."""
        if not self.memory_manager or not self.memory_manager.memory:
            return {"memory_enabled": False}

        session_ctx = self.memory_manager.get_session_context(self.identity_id)
        return {
            "memory_enabled": True,
            "interactions_stored_session": len(session_ctx),
        }

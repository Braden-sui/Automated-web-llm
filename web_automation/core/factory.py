"""Factory functions for creating browser agents."""
from typing import Optional, Dict, Any

from web_automation.core.browser_agent import PlaywrightBrowserAgent


def create_playwright_agent(memory_enabled: bool = False, **kwargs) -> PlaywrightBrowserAgent:
    """
    Factory function to create a browser agent with optional memory enhancement.

    Args:
        memory_enabled: If True, returns a PersistentMemoryBrowserAgent.
        **kwargs: Additional arguments to pass to the agent constructor.

    Returns:
        An instance of PlaywrightBrowserAgent or PersistentMemoryBrowserAgent.
    """
    if memory_enabled:
        from web_automation.memory.memory_enhanced_agent import PersistentMemoryBrowserAgent # Deferred import
        return PersistentMemoryBrowserAgent(**kwargs)
    return PlaywrightBrowserAgent(**kwargs)

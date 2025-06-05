import logging
import re
import asyncio
import random
from typing import Optional

from ..core.browser_agent import PlaywrightBrowserAgent, BrowserAgentError
from .memory_manager import Mem0BrowserAdapter
from ..config.config_models import Mem0AdapterConfig

logger = logging.getLogger(__name__)

class PersistentMemoryBrowserAgent(PlaywrightBrowserAgent):
    """PlaywrightBrowserAgent with additional helpers that utilise the Mem0BrowserAdapter."""

    def __init__(self, dependencies=None, mem0_config: Optional[Mem0AdapterConfig] = None, **kwargs):
        # Handle both new DI pattern and legacy pattern
        if dependencies is not None:
            # New DI pattern - memory_manager is already created by BrowserAgentFactory
            super().__init__(dependencies=dependencies)
            # In this case, self.memory_manager and self.memory_enabled were already set by PlaywrightBrowserAgent.__init__
            # Just log that we're using the DI pattern
            logger.info("PersistentMemoryBrowserAgent: Initialized using dependency injection pattern")
            return
            
        # Legacy pattern - the recent fix for memory initialization
        # Tell the parent class NOT to initialize memory, as we will handle it
        kwargs_for_super = kwargs.copy()
        kwargs_for_super["memory_enabled"] = False
        super().__init__(**kwargs_for_super)

        # Now, this agent is responsible for memory
        self.memory_enabled = True

        # Get the appropriate memory config
        config_to_use = mem0_config
        memory_config_dict = kwargs.get('memory_config')
        
        # If memory_config was passed as a dict (from create_playwright_agent)
        if memory_config_dict and not config_to_use:
            from ..config.config_models import Mem0AdapterConfig
            try:
                config_to_use = Mem0AdapterConfig(**memory_config_dict)
                logger.info("PersistentMemoryBrowserAgent: Created Mem0AdapterConfig from memory_config dict")
            except Exception as e:
                logger.error(f"PersistentMemoryBrowserAgent: Failed to create Mem0AdapterConfig from dict: {e}")
        
        # If no config was provided anywhere, use the global default
        if config_to_use is None:
            from ..config.settings import mem0_adapter_config as global_mem0_config
            config_to_use = global_mem0_config
            logger.info("PersistentMemoryBrowserAgent: Using global default Mem0AdapterConfig")

        try:
            self.memory_manager = Mem0BrowserAdapter(mem0_config=config_to_use)
            collection_name = config_to_use.qdrant_collection_name if config_to_use else "default (Memory())"
            logger.info(f"PersistentMemoryBrowserAgent: Mem0BrowserAdapter initialized. Collection: {collection_name}")
        except Exception as e:
            logger.error(f"PersistentMemoryBrowserAgent: Failed to initialize Mem0BrowserAdapter: {e}")
            self.memory_manager = None
            self.memory_enabled = False  # Memory initialization failed

    async def smart_selector_click(self, target_description: str, fallback_selector: str, timeout: int = 5000) -> bool:
        """Attempt click using selectors learned from memory before falling back."""
        print(f"\n=== SMART_SELECTOR_CLICK DEBUG START ===")
        print(f"Target: '{target_description}'")
        print(f"Fallback: '{fallback_selector}'")
        print(f"Memory manager available: {self.memory_manager is not None}")
        selectors_to_try = []

        if self.memory_manager:
            results = self.memory_manager.search_automation_patterns(
                pattern_query=target_description,
                user_id=self.identity_id,
                limit=3,
            )
            for res in results:
                retrieved_selector = res.get("metadata", {}).get("selector")
                if retrieved_selector:
                    selectors_to_try.append(retrieved_selector)

        if self.memory_manager:
            print(f"Memory search completed. Found {len(results)} results:")
            for i, res in enumerate(results):
                print(f"  Result {i}: memory='{res.get('memory', '')}', metadata='{res.get('metadata', {})}'")
        else:
            print("No memory manager - skipping memory search")
        selectors_to_try.append(fallback_selector)
        print(f"Selectors to try: {selectors_to_try}")

        for sel in selectors_to_try:
            print(f"Trying selector: '{sel}'")
            try:
                element = await self._get_element(sel, timeout)
                await asyncio.sleep(random.uniform(0.1, 0.3)) # Human-like delay before click
                await element.click()
                print(f"SUCCESS with selector: '{sel}'")
                print(f"About to store pattern in memory...")
                if self.memory_manager:
                    self.memory_manager.store_automation_pattern(description=target_description, selector=sel, success=True, user_id=self.identity_id)
                    print(f"Pattern storage attempted")
                return True
            except Exception as e:
                logger.debug(f"Selector {sel} failed: {e}")
                continue

        if self.memory_manager:
            # When all selectors fail, we might want to record this failure against the target_description.
            # Storing the fallback_selector as the 'failed' selector could be one approach, or None if no specific selector led to this final failure.
            self.memory_manager.store_automation_pattern(
                description=target_description, 
                selector=fallback_selector, # Or perhaps a generic 'NO_SELECTOR_WORKED' or the last attempted 'sel'
                success=False, 
                user_id=self.identity_id
            )
        print(f"=== SMART_SELECTOR_CLICK DEBUG END ===\n")
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

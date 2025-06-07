from typing import Optional, Dict
from .dependencies import BrowserAgentFactory

def create_playwright_agent(memory_enabled: bool = False, reasoning_enabled: bool = None, memory_config: Optional[Dict] = None, **kwargs):
    # Handle reasoning configuration
    if reasoning_enabled is not None:
        from ..config.settings import reasoning_config
        reasoning_config.enabled = reasoning_enabled
    
    if memory_enabled:
        final_memory_config = {'enabled': True}
        if memory_config:
            # Ensure memory_config is converted to a dict if it's a Pydantic model
            # The type hint for memory_config is Optional[Dict], but it receives a Mem0AdapterConfig object from the test.
            if hasattr(memory_config, 'model_dump'): # Check if it's a Pydantic model (or similar)
                final_memory_config.update(memory_config.model_dump())
            elif isinstance(memory_config, dict):
                final_memory_config.update(memory_config)
            else:
                # Log a warning or raise an error if the type is unexpected
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Unexpected type for memory_config in create_playwright_agent: {type(memory_config)}. Expected Dict or Pydantic model.")
        return BrowserAgentFactory.create_agent(memory_config=final_memory_config, **kwargs)
    else:
        return BrowserAgentFactory.create_agent(**kwargs)

from typing import Optional, Dict
from .dependencies import BrowserAgentFactory

def create_playwright_agent(memory_enabled: bool = False, memory_config: Optional[Dict] = None, **kwargs):
    if memory_enabled:
        final_memory_config = {'enabled': True}
        if memory_config:
            final_memory_config.update(memory_config)
        return BrowserAgentFactory.create_agent(memory_config=final_memory_config, **kwargs)
    else:
        return BrowserAgentFactory.create_agent(**kwargs)

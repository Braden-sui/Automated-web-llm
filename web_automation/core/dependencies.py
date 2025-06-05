from typing import Protocol, Optional, Dict, Any
from dataclasses import dataclass

class MemoryManagerProtocol(Protocol):
    def store_automation_pattern(self, pattern: str, success: bool, user_id: str, metadata: dict = None) -> None: ...
    def search_automation_patterns(self, pattern_query: str, user_id: str, limit: int = 5) -> list: ...
    async def store_session_context(self, user_id: str, context_data: dict, metadata: dict = None) -> None: ...
    async def search_session_context(self, user_id: str, query: str, limit: int = 5) -> list: ...

@dataclass
class BrowserAgentDependencies:
    memory_manager: Optional[MemoryManagerProtocol] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}

class DependencyFactory:
    @staticmethod
    def create_memory_manager(config: Optional[Dict] = None) -> Optional[MemoryManagerProtocol]:
        if not config or not config.get('enabled', True):
            return None
        try:
            from web_automation.memory.memory_manager import Mem0BrowserAdapter
            from web_automation.config.config_models import Mem0AdapterConfig
            mem0_config = Mem0AdapterConfig(**config) if isinstance(config, dict) else config
            return Mem0BrowserAdapter(mem0_config=mem0_config)
        except Exception:
            return None

class BrowserAgentFactory:
    @staticmethod
    def create_agent(memory_config: Optional[Dict] = None, **kwargs):
        full_config = {
            'identity_id': kwargs.get('identity_id', 'default'),
            'headless': kwargs.get('headless', True),
            **kwargs
        }
        
        deps = BrowserAgentDependencies(config=full_config)
        
        if memory_config and memory_config.get('enabled'):
            deps.memory_manager = DependencyFactory.create_memory_manager(memory_config)
        
        if memory_config:
            from web_automation.memory.memory_enhanced_agent import PersistentMemoryBrowserAgent
            return PersistentMemoryBrowserAgent(deps)
        else:
            from web_automation.core.browser_agent import PlaywrightBrowserAgent
            return PlaywrightBrowserAgent(deps)

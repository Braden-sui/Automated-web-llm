from typing import Protocol, Optional, Dict, Any
from dataclasses import dataclass
import logging # Added import
from pydantic import ValidationError # Added import

logger = logging.getLogger(__name__) # Added logger instance

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
            logger.debug("Memory manager explicitly disabled or no config provided.")
            return None
        
        try:
            from web_automation.memory.memory_manager import Mem0BrowserAdapter
            from web_automation.config.config_models import Mem0AdapterConfig
            
            logger.debug(f"Attempting to create memory manager with raw config: {config}")
            logger.debug(f"Config type: {type(config)}")
            logger.debug(f"Config keys: {list(config.keys()) if isinstance(config, dict) else 'Not a dict'}")
            
            if isinstance(config, dict) and 'enabled' in config and len(config) == 1:
                logger.error("Config only contains 'enabled' key - this often indicates a Pydantic serialization failure upstream. Aborting memory manager creation.")
                return None
            
            # Ensure we are working with a Mem0AdapterConfig instance
            # If config is already a Mem0AdapterConfig, use it directly.
            # If it's a dict, try to parse it into Mem0AdapterConfig.
            if isinstance(config, Mem0AdapterConfig):
                mem0_adapter_instance_config = config
                logger.debug("Provided config is already a Mem0AdapterConfig instance.")
            elif isinstance(config, dict):
                logger.debug("Provided config is a dict, attempting to parse into Mem0AdapterConfig.")
                mem0_adapter_instance_config = Mem0AdapterConfig(**config)
            else:
                logger.error(f"Invalid config type for memory manager: {type(config)}. Expected Dict or Mem0AdapterConfig.")
                return None

            logger.info(f"Successfully prepared Mem0AdapterConfig: {mem0_adapter_instance_config.model_dump_json(indent=2)}")
            return Mem0BrowserAdapter(mem0_config=mem0_adapter_instance_config)
            
        except ValidationError as e:
            logger.error(f"Mem0AdapterConfig validation failed during memory manager creation: {e}")
            for error in e.errors():
                logger.error(f"  Field: {error['loc']}, Type: {error['type']}, Message: {error['msg']}")
            return None
        except ImportError as e:
            logger.error(f"Import error during memory manager creation (Mem0BrowserAdapter or Mem0AdapterConfig not found?): {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating memory manager: {e}", exc_info=True)
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

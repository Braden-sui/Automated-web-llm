from typing import Protocol, Optional, Dict, Any
from dataclasses import dataclass, field
import logging
from pydantic import ValidationError
import ollama
from web_automation.config.config_models import VisualSystemConfig # Import VisualSystemConfig at the top

logger = logging.getLogger(__name__)

class MemoryManagerProtocol(Protocol):
    def store_automation_pattern(self, pattern: str, success: bool, user_id: str, metadata: dict = None) -> None: ...
    def search_automation_patterns(self, pattern_query: str, user_id: str, limit: int = 5) -> list: ...
    async def store_session_context(self, user_id: str, context_data: dict, metadata: dict = None) -> None: ...
    async def search_session_context(self, user_id: str, query: str, limit: int = 5) -> list: ...
    def store_visual_pattern(self, user_id: str, description: str, visual_data: dict, metadata: Optional[dict] = None) -> None: ...
    def search_visual_patterns(self, query_description: str, user_id: str, limit: int = 5) -> list: ...

@dataclass
class BrowserAgentDependencies:
    memory_manager: Optional[MemoryManagerProtocol] = None
    config: Dict[str, Any] = field(default_factory=dict)
    ollama_client: Optional[ollama.AsyncClient] = None
    visual_llm_model_name: Optional[str] = None
    
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
            print(f"--- DEP_FACTORY: create_memory_manager trying... Config: {config}")
            from web_automation.memory.memory_manager import Mem0BrowserAdapter
            from web_automation.config.config_models import Mem0AdapterConfig
            print(f"--- DEP_FACTORY: Imports successful.")
            
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
                print("--- DEP_FACTORY: Config is Mem0AdapterConfig instance.")
            elif isinstance(config, dict):
                logger.debug("Provided config is a dict, attempting to parse into Mem0AdapterConfig.")
                print("--- DEP_FACTORY: Config is dict, parsing to Mem0AdapterConfig...")
                mem0_adapter_instance_config = Mem0AdapterConfig(**config)
                print("--- DEP_FACTORY: Parsed dict to Mem0AdapterConfig.")
            else:
                logger.error(f"Invalid config type for memory manager: {type(config)}. Expected Dict or Mem0AdapterConfig.")
                print(f"--- DEP_FACTORY: Invalid config type: {type(config)}.")
                return None

            logger.info(f"Successfully prepared Mem0AdapterConfig: {mem0_adapter_instance_config.model_dump_json(indent=2)}")
            print(f"--- DEP_FACTORY: Mem0AdapterConfig prepared. Attempting to create Mem0BrowserAdapter...")
            adapter = Mem0BrowserAdapter(mem0_config=mem0_adapter_instance_config)
            print(f"--- DEP_FACTORY: Mem0BrowserAdapter CREATED: {type(adapter).__name__}")
            return adapter
            
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
            print(f"--- DEP_FACTORY: EXCEPTION in create_memory_manager: {e}")
            import traceback
            traceback.print_exc()
            return None

from web_automation.config.config_models import VisualSystemConfig # Import VisualSystemConfig

class BrowserAgentFactory:
    @staticmethod
    def create_agent(memory_config: Optional[Dict] = None, visual_config_input: Optional[Dict] = None, **kwargs):
        full_config = {
            'identity_id': kwargs.get('identity_id', 'default_agent'),
            'headless': kwargs.get('headless', True),
            **kwargs
        }

        print(f"--- FACTORY: create_agent for {full_config['identity_id']} starting ---")
        deps = BrowserAgentDependencies(config=full_config)
        print(f"--- FACTORY: BrowserAgentDependencies created for {full_config['identity_id']} ---")

        # --- Visual System: Default to DISABLED unless explicitly enabled ---
        # If visual_config_input is None, or does not specify 'enabled', force enabled=False
        if visual_config_input is None:
            visual_config_input = {'enabled': False, 'auto_capture': False}
        elif isinstance(visual_config_input, dict):
            visual_config_input = dict(visual_config_input)  # Defensive copy
            if 'enabled' not in visual_config_input:
                visual_config_input['enabled'] = False
            if 'auto_capture' not in visual_config_input:
                visual_config_input['auto_capture'] = False

        # The rest of the function remains unchanged below this point

        # Process Memory Configuration
        memory_enabled = False
        if memory_config and memory_config.get('enabled', False):
            print(f"--- FACTORY: Attempting to create memory manager for {full_config['identity_id']} ---")
            deps.memory_manager = DependencyFactory.create_memory_manager(memory_config)
            if deps.memory_manager:
                memory_enabled = True
                print(f"--- FACTORY: Memory manager CREATED for {full_config['identity_id']}: {type(deps.memory_manager).__name__} ---")
            else:
                logger.warning(f"Memory manager creation failed for agent {full_config['identity_id']}, though enabled. Proceeding without memory.")

        # Process Visual Configuration
        visual_system_config_actual: Optional[VisualSystemConfig] = None
        if isinstance(visual_config_input, VisualSystemConfig):
            visual_system_config_actual = visual_config_input
        elif isinstance(visual_config_input, dict):
            try:
                visual_system_config_actual = VisualSystemConfig(**visual_config_input)
            except ValidationError as e:
                logger.error(f"VisualSystemConfig validation failed: {e}. Visual features will be disabled.")
                visual_system_config_actual = VisualSystemConfig(enabled=False) # Default to disabled
        else:
            visual_system_config_actual = VisualSystemConfig(enabled=False) # Default to disabled if no input

        visual_enabled = visual_system_config_actual.enabled
        if visual_enabled:
            try:
                ollama_base_url = visual_system_config_actual.ollama_base_url
                if ollama_base_url:
                    deps.ollama_client = ollama.AsyncClient(host=ollama_base_url)
                    logger.info(f"Initialized Ollama AsyncClient with base URL: {ollama_base_url} for agent {full_config['identity_id']}")
                else:
                    deps.ollama_client = ollama.AsyncClient()
                    logger.info(f"Initialized Ollama AsyncClient with default host for agent {full_config['identity_id']}.")
                print(f"--- FACTORY: Ollama client for visual features INITIALIZED for {full_config['identity_id']}: {type(deps.ollama_client).__name__} ---")
                deps.visual_llm_model_name = visual_system_config_actual.model_name
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client for visual capabilities for agent {full_config['identity_id']}: {e}. Visual features disabled.")
                deps.ollama_client = None
                deps.visual_llm_model_name = None
                visual_enabled = False # Update status
        else:
            print(f"--- FACTORY: Ollama client for visual features is None for {full_config['identity_id']}. Visual features remain disabled. ---")

        print(f"--- FACTORY: Determining agent type for {full_config['identity_id']} (memory: {memory_enabled}, visual: {visual_enabled and deps.ollama_client is not None}) ---")
        # Determine agent type based on successfully initialized capabilities
        if memory_enabled or (visual_enabled and deps.ollama_client):
            from web_automation.memory.memory_enhanced_agent import PersistentMemoryBrowserAgent
            logger.info(f"Creating PersistentMemoryBrowserAgent for {full_config['identity_id']} (memory: {memory_enabled}, visual: {visual_enabled and deps.ollama_client is not None})")
            print(f"--- FACTORY: Returning PersistentMemoryBrowserAgent for {full_config['identity_id']} ---")
            # Always pass visual_config_input as a dict
            visual_config_dict = visual_system_config_actual.model_dump() if hasattr(visual_system_config_actual, 'model_dump') else visual_system_config_actual
            return PersistentMemoryBrowserAgent(dependencies=deps, visual_config_input=visual_config_dict)
        else:
            from web_automation.core.browser_agent import PlaywrightBrowserAgent
            logger.info(f"Creating PlaywrightBrowserAgent for {full_config['identity_id']} (no advanced features enabled/initialized).")
            print(f"--- FACTORY: Returning PlaywrightBrowserAgent for {full_config['identity_id']} ---")
            return PlaywrightBrowserAgent(dependencies=deps)

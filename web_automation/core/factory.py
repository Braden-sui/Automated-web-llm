from typing import Optional, Dict, Any, List
from .browser_agent import PlaywrightBrowserAgent
from .dependencies import BrowserAgentDependencies
from web_automation.config.settings import browser_config, visual_system_config, captcha_config
from web_automation.config.config_models import Mem0AdapterConfig, VisualSystemConfig, BrowserConfig
import logging

logger = logging.getLogger(__name__)

class BrowserAgentFactory:
    @staticmethod
    def create_agent(
        memory_config: Optional[Dict] = None,
        visual_config_input: Optional[Dict] = None,
        browser_config_input: Optional[Dict] = None,
        plugins: Optional[List[Any]] = None,
        **kwargs
    ) -> PlaywrightBrowserAgent:
        """
        Create a Playwright agent with the specified configurations and plugins.

        Args:
            memory_config: Configuration for memory system.
            visual_config_input: Configuration for visual system.
            browser_config_input: Configuration for browser settings.
            plugins: List of plugins to attach to the agent.
            **kwargs: Additional arguments for agent configuration.

        Returns:
            PlaywrightBrowserAgent: Configured agent instance.
        """
        full_config = {
            'identity_id': kwargs.get('identity_id', 'default_agent'),
            'headless': kwargs.get('headless', True),
            **kwargs
        }

        logger.info(f"Creating agent for {full_config['identity_id']}")
        deps = BrowserAgentDependencies(config=full_config)
        agent = PlaywrightBrowserAgent(dependencies=deps)

        # Initialize plugins
        if plugins is None:
            plugins = []
            # Add default plugins based on configuration
            if memory_config and memory_config.get('enabled', True):
                from web_automation.memory.memory_plugin import MemoryPlugin
                plugins.append(MemoryPlugin())
            if visual_config_input and visual_config_input.get('enabled', False):
                from web_automation.vision.visual_memory_plugin import VisualMemoryPlugin
                plugins.append(VisualMemoryPlugin())
            from web_automation.captcha.captcha_plugin import CaptchaPlugin
            plugins.append(CaptchaPlugin())

        agent.plugins = {}
        for plugin in plugins:
            plugin.initialize(agent)
            agent.plugins[plugin.get_name()] = plugin
            logger.info(f"Plugin {plugin.get_name()} attached to agent {agent.identity_id}")

        logger.info(f"Agent {agent.identity_id} created with {len(plugins)} plugins")
        return agent

def create_playwright_agent(memory_enabled: bool = False, memory_config: Optional[Dict] = None, **kwargs):
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

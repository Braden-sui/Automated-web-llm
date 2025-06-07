from typing import Any, Optional, Dict, List
from web_automation.core.plugin_base import AgentPlugin
from web_automation.memory.memory_manager import Mem0BrowserAdapter
import logging

logger = logging.getLogger(__name__)

class MemoryPlugin(AgentPlugin):
    """
    Plugin for handling AI memory capabilities in the web automation framework using Mem0.
    """
    def __init__(self, memory_manager: Optional[Mem0BrowserAdapter] = None):
        self.memory_manager = memory_manager
        self.agent = None

    def initialize(self, agent: Any) -> None:
        """
        Initialize the Memory plugin with the agent instance.

        Args:
            agent: The agent instance to which this plugin is attached.
        """
        self.agent = agent
        if self.memory_manager is None:
            from web_automation.config.config_models import Mem0AdapterConfig
            mem0_config = Mem0AdapterConfig(
                agent_id=agent.identity_id if hasattr(agent, 'identity_id') else None
            )
            self.memory_manager = Mem0BrowserAdapter(mem0_config)
        logger.info("Memory Plugin initialized for agent")

    def get_name(self) -> str:
        """
        Return the name of the plugin.

        Returns:
            str: The name of the plugin.
        """
        return "memory"

    def is_enabled(self) -> bool:
        """
        Check if the memory system is enabled.

        Returns:
            bool: True if enabled, False otherwise.
        """
        return self.memory_manager is not None

    async def store_automation_pattern(self, pattern_data: Dict, success: bool) -> None:
        """
        Store an automation pattern in memory.

        Args:
            pattern_data: Dictionary containing pattern data.
            success: Boolean indicating if the action was successful.
        """
        if not self.is_enabled():
            logger.warning("Memory system is not enabled, skipping store")
            return
        try:
            await self.memory_manager.store_automation_pattern(pattern_data, success)
            if hasattr(self.agent, 'interactions_stored_session'):
                self.agent.interactions_stored_session += 1
        except Exception as e:
            logger.error(f"Error storing automation pattern: {e}", exc_info=True)

    async def search_similar_patterns(self, query_data: Dict, limit: int = 3) -> List[Dict]:
        """
        Search for similar automation patterns in memory.

        Args:
            query_data: Dictionary containing query parameters.
            limit: Maximum number of results to return.

        Returns:
            List[Dict]: List of similar pattern data.
        """
        if not self.is_enabled():
            logger.warning("Memory system is not enabled, skipping search")
            return []
        try:
            return await self.memory_manager.search_similar_patterns(query_data, limit)
        except Exception as e:
            logger.error(f"Error searching for similar patterns: {e}", exc_info=True)
            return []

    def get_memory_stats(self) -> Dict:
        """
        Get statistics about the memory system.

        Returns:
            Dict: Dictionary containing memory statistics.
        """
        if not self.is_enabled():
            return {"memory_enabled": False}
        try:
            stats = {"memory_enabled": True}
            if hasattr(self.agent, 'interactions_stored_session'):
                stats["interactions_stored_session"] = self.agent.interactions_stored_session
            return stats
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}", exc_info=True)
            return {"memory_enabled": False}

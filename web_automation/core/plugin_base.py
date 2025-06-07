from abc import ABC, abstractmethod
from typing import Any

class AgentPlugin(ABC):
    """
    Abstract base class for agent plugins, defining the interface for extending agent capabilities.
    """

    @abstractmethod
    def initialize(self, agent: Any) -> None:
        """
        Initialize the plugin with the agent instance.

        Args:
            agent: The agent instance to which this plugin is attached.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the plugin.

        Returns:
            str: The name of the plugin.
        """
        pass

    def on_page_load(self) -> None:
        """
        Optional hook called when a new page is loaded.
        Can be overridden by plugins to perform actions on page load.
        """
        pass

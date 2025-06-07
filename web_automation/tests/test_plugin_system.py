import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from web_automation.core.plugin_base import AgentPlugin
from web_automation.core.browser_agent import PlaywrightBrowserAgent
from web_automation.core.factory import BrowserAgentFactory

# Mock Plugin for Testing
class MockPlugin(AgentPlugin):
    def __init__(self, name="mock_plugin"):
        self._name = name
        self.agent = None

    def initialize(self, agent):
        self.agent = agent

    def get_name(self):
        return self._name

@pytest.fixture
async def mock_agent():
    # Create a mock dependencies object
    mock_deps = MagicMock()
    mock_deps.config = {"identity_id": "test_agent", "headless": True}
    
    # Instantiate the agent
    agent = PlaywrightBrowserAgent(mock_deps)
    agent.start = AsyncMock()
    agent.shutdown = AsyncMock()
    
    yield agent
    
    # Cleanup if necessary
    await agent.shutdown()

@pytest.mark.asyncio
async def test_plugin_initialization(mock_agent):
    # Create mock plugins
    plugin1 = MockPlugin("plugin1")
    plugin2 = MockPlugin("plugin2")
    plugins = [plugin1, plugin2]

    # Use factory to create agent with plugins
    factory = BrowserAgentFactory()
    factory.create_agent = MagicMock(return_value=mock_agent)
    agent = await factory.create_agent_with_plugins(plugins=plugins)

    # Verify plugins are initialized and attached to agent
    assert len(agent.plugins) == 2
    assert agent.plugins["plugin1"] == plugin1
    assert agent.plugins["plugin2"] == plugin2
    assert plugin1.agent == agent
    assert plugin2.agent == agent

@pytest.mark.asyncio
async def test_plugin_access_during_execution(mock_agent):
    # Create a mock plugin
    plugin = MockPlugin("test_plugin")
    plugins = [plugin]

    # Use factory to create agent with plugins
    factory = BrowserAgentFactory()
    factory.create_agent = MagicMock(return_value=mock_agent)
    agent = await factory.create_agent_with_plugins(plugins=plugins)

    # Simulate an instruction execution that accesses a plugin
    mock_instruction = MagicMock()
    mock_instruction.action_type = "CLICK"
    agent.executors["CLICK"].execute = AsyncMock(return_value=True)

    # Execute instruction
    result = await agent.execute_instruction_with_state_management(mock_instruction)

    # Verify plugin is accessible
    assert "test_plugin" in agent.plugins
    assert agent.plugins["test_plugin"].get_name() == "test_plugin"
    assert result == True

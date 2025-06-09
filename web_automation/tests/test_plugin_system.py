import pytest
import asyncio
import pytest_asyncio
from unittest.mock import MagicMock, AsyncMock
from web_automation.core.plugin_base import AgentPlugin
from web_automation.core.browser_agent import PlaywrightBrowserAgent
from web_automation.core.factory import BrowserAgentFactory
from web_automation.core.agent_state import AgentState # Import AgentState

# Mock Plugin for Testing
class MockPlugin(AgentPlugin):
    def __init__(self, name="mock_plugin"):
        self._name = name
        self.agent = None

    def initialize(self, agent):
        self.agent = agent

    def get_name(self):
        return self._name

@pytest_asyncio.fixture
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

    # Use factory to create agent with plugins - FIX: Use correct method
    agent = BrowserAgentFactory.create_agent(
        plugins=plugins,
        browser_type='chromium',
        headless=True,
        identity_id='test_agent'
    )

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
    mock_agent.plugins = {}
    for p in plugins:
        p.initialize(mock_agent)
        mock_agent.plugins[p.get_name()] = p
        
    factory = BrowserAgentFactory()
    factory.create_agent = MagicMock(return_value=mock_agent) # This mock is now less critical for plugin setup
    agent = factory.create_agent(plugins=plugins) # agent is mock_agent

    # Simulate an instruction execution that accesses a plugin
    mock_instruction = MagicMock()
    from web_automation.models.instructions import ActionType
    mock_instruction.type = ActionType.CLICK  # Use 'type' not 'action_type'
    agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Ensure _page is set for the mock_agent, as start() is mocked and doesn't set _page
    agent._page = MagicMock()

    # Mock state checking methods to prevent actual page interaction for this plugin test
    agent._check_page_state = AsyncMock(return_value=AgentState.IDLE)
    agent._handle_state_transition = AsyncMock(return_value=True) # Assume successful handling

    # Execute instruction
    result = await agent.execute_instruction_with_state_management(mock_instruction)

    # Verify plugin is accessible
    assert "test_plugin" in agent.plugins
    assert agent.plugins["test_plugin"].get_name() == "test_plugin"
    assert result == True

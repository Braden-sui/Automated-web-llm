import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock
from web_automation.core.browser_agent import PlaywrightBrowserAgent
from web_automation.core.agent_state import AgentState
from web_automation.models.instructions import ClickInstruction, ActionType

@pytest_asyncio.fixture
async def mock_agent():
    # Create a mock dependencies object
    mock_deps = MagicMock()
    mock_deps.config = {"identity_id": "test_agent", "headless": True}
    
    # Instantiate the agent
    agent = PlaywrightBrowserAgent(mock_deps)
    agent.start = AsyncMock()
    agent.shutdown = AsyncMock()
    agent._page = MagicMock()
    
    yield agent
    
    # Cleanup if necessary
    await agent.shutdown()

@pytest.mark.asyncio
async def test_state_transition_to_captcha_required(mock_agent):
    # Mock the page state check to return CAPTCHA_REQUIRED
    mock_agent._check_page_state = AsyncMock(return_value=AgentState.CAPTCHA_REQUIRED)
    mock_agent._handle_captcha_state = AsyncMock(return_value=True)

    # Create a mock instruction
    instruction = ClickInstruction(selector=".button")
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Execute the instruction with state management
    result = await mock_agent.execute_instruction_with_state_management(instruction)

    # Verify state transition and handling
    assert mock_agent.current_state == AgentState.RECOVERING
    mock_agent._check_page_state.assert_called()
    mock_agent._handle_captcha_state.assert_called_once()

@pytest.mark.asyncio
async def test_state_transition_to_unexpected_modal(mock_agent):
    # Mock the page state check to return UNEXPECTED_MODAL
    mock_agent._check_page_state = AsyncMock(return_value=AgentState.UNEXPECTED_MODAL)
    mock_agent._handle_modal_state = AsyncMock(return_value=True)

    # Create a mock instruction
    instruction = ClickInstruction(selector=".button")
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Execute the instruction with state management
    result = await mock_agent.execute_instruction_with_state_management(instruction)

    # Verify state transition and handling
    assert mock_agent.current_state == AgentState.RECOVERING
    mock_agent._check_page_state.assert_called()
    mock_agent._handle_modal_state.assert_called_once()

@pytest.mark.asyncio
async def test_state_transition_to_fatal_error(mock_agent):
    # Mock the page state check to return FATAL_ERROR
    mock_agent._check_page_state = AsyncMock(return_value=AgentState.FATAL_ERROR)
    mock_agent._handle_state_transition = AsyncMock(return_value=False)

    # Create a mock instruction
    instruction = ClickInstruction(selector=".button")
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Execute the instruction with state management and expect an exception
    with pytest.raises(RuntimeError, match="Fatal error detected before executing instruction"):
        await mock_agent.execute_instruction_with_state_management(instruction)

    # Verify state transition
    assert mock_agent.current_state == AgentState.FATAL_ERROR
    mock_agent._check_page_state.assert_called()

@pytest.mark.asyncio
async def test_state_recovery_after_captcha(mock_agent):
    # Simulate CAPTCHA state and successful recovery
    mock_agent._check_page_state = AsyncMock(side_effect=[AgentState.CAPTCHA_REQUIRED, AgentState.EXECUTING])
    mock_agent._handle_captcha_state = AsyncMock(return_value=True)

    # Create a mock instruction
    instruction = ClickInstruction(selector=".button")
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Execute the instruction with state management
    result = await mock_agent.execute_instruction_with_state_management(instruction)

    # Verify state transition and recovery
    assert mock_agent.current_state == AgentState.EXECUTING
    assert result == True
    mock_agent._check_page_state.assert_called()
    mock_agent._handle_captcha_state.assert_called_once()
    mock_agent.executors[ActionType.CLICK].execute.assert_called_once()

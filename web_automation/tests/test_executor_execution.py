import pytest
import pytest_asyncio
import asyncio
from unittest.mock import MagicMock, AsyncMock
from web_automation.core.browser_agent import PlaywrightBrowserAgent
from web_automation.core.dependencies import BrowserAgentDependencies
from web_automation.core.agent_state import AgentState
from web_automation.models.instructions import ClickInstruction, TypeInstruction, NavigateInstruction, ActionType

@pytest_asyncio.fixture
async def mock_agent():
    # Create proper dependencies
    dependencies = BrowserAgentDependencies(
        config={"identity_id": "test_agent", "headless": True}
    )
    
    # Instantiate the agent with proper dependencies
    agent = PlaywrightBrowserAgent(dependencies)
    
    # Mock the page and browser infrastructure
    agent._page = AsyncMock()
    agent._context = AsyncMock()
    agent._browser = AsyncMock()
    agent._playwright = AsyncMock()
    
    # Mock state management methods
    agent.current_state = AgentState.EXECUTING
    agent._check_page_state = AsyncMock(return_value=AgentState.EXECUTING)
    agent._handle_state_transition = AsyncMock(return_value=True)
    
    yield agent
    
    # Cleanup - mock the close method
    agent.close = AsyncMock()
    await agent.close()

@pytest.mark.asyncio
async def test_click_executor_execution(mock_agent):
    # Mock the execute method of the ClickExecutor
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)

    # Create a click instruction
    instruction = ClickInstruction(selector=".button")
    
    # Execute the instruction
    result = await mock_agent.execute_instruction_with_state_management(instruction)
    
    # Verify the result
    assert result == True
    mock_agent.executors[ActionType.CLICK].execute.assert_called_once_with(mock_agent._page, instruction)

@pytest.mark.asyncio
async def test_type_executor_execution(mock_agent):
    # Mock the execute method of the TypeExecutor
    mock_agent.executors[ActionType.TYPE].execute = AsyncMock(return_value=True)

    # Create a type instruction
    instruction = TypeInstruction(selector="input", text="test input")
    
    # Execute the instruction
    result = await mock_agent.execute_instruction_with_state_management(instruction)
    
    # Verify the result
    assert result == True
    mock_agent.executors[ActionType.TYPE].execute.assert_called_once_with(mock_agent._page, instruction)

@pytest.mark.asyncio
async def test_navigate_executor_execution(mock_agent):
    # Mock the execute method of the NavigateExecutor
    mock_agent.executors[ActionType.NAVIGATE].execute = AsyncMock(return_value=True)

    # Create a navigate instruction
    instruction = NavigateInstruction(url="https://example.com")
    
    # Execute the instruction
    result = await mock_agent.execute_instruction_with_state_management(instruction)
    
    # Verify the result
    assert result == True
    mock_agent.executors[ActionType.NAVIGATE].execute.assert_called_once_with(mock_agent._page, instruction)

@pytest.mark.asyncio
async def test_multiple_instructions_execution(mock_agent):
    # Mock the execute methods for different executors
    mock_agent.executors[ActionType.CLICK].execute = AsyncMock(return_value=True)
    mock_agent.executors[ActionType.TYPE].execute = AsyncMock(return_value=True)
    mock_agent.executors[ActionType.NAVIGATE].execute = AsyncMock(return_value=True)

    # Create a list of instructions
    instructions = [
        ClickInstruction(selector=".button"),
        TypeInstruction(selector="input", text="test input"),
        NavigateInstruction(url="https://example.com")
    ]
    
    # Execute the instructions
    results = await mock_agent.execute_instructions(instructions)
    
    # Verify the results
    assert len(results) == 3
    assert results == [True, True, True]
    mock_agent.executors[ActionType.CLICK].execute.assert_called_once()
    mock_agent.executors[ActionType.TYPE].execute.assert_called_once()
    mock_agent.executors[ActionType.NAVIGATE].execute.assert_called_once()

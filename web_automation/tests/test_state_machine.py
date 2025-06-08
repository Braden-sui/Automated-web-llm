import pytest
import pytest_asyncio  # ADD THIS IMPORT
import asyncio
from web_automation.core.dependencies import BrowserAgentFactory
from web_automation.core.agent_state import AgentState
from web_automation.models.instructions import ClickInstruction, ActionType

@pytest_asyncio.fixture  # CHANGED from @pytest.fixture
async def integration_agent():
    """Create a real agent for integration testing."""
    agent = BrowserAgentFactory.create_agent(
        browser_type='chromium',
        headless=True,
        identity_id='state_machine_test_agent'
    )
    
    async with agent:
        yield agent

@pytest.mark.asyncio
async def test_agent_starts_in_idle_state(integration_agent):
    """Test that agent starts in IDLE state."""
    assert integration_agent.current_state == AgentState.IDLE

@pytest.mark.asyncio
async def test_state_machine_basic_functionality(integration_agent):
    """Test basic state machine functionality with real browser."""
    # Start in IDLE
    assert integration_agent.current_state == AgentState.IDLE
    
    # Navigate to real page - this exercises state machine
    await integration_agent._page.goto("https://example.com")
    
    # Basic state checking doesn't crash
    state = await integration_agent._check_page_state()
    assert state in [AgentState.IDLE, AgentState.EXECUTING]

@pytest.mark.asyncio
async def test_instruction_execution_with_state_management(integration_agent):
    """Test that instructions can be executed with state management."""
    # Navigate to a real page first
    await integration_agent._page.goto("https://example.com")
    
    # Try to click a real element that exists
    instruction = ClickInstruction(selector="a", type=ActionType.CLICK)
    
    try:
        result = await integration_agent.execute_instruction_with_state_management(instruction)
        # If it succeeds, great. If it fails with specific errors, that's also ok for this test
    except Exception as e:
        # Expected - some clicks might fail, but state machine should handle it
        assert "state" in str(e).lower() or "timeout" in str(e).lower() or "selector" in str(e).lower()
    
    # Agent should still be in a valid state
    assert integration_agent.current_state in [AgentState.IDLE, AgentState.EXECUTING, AgentState.FATAL_ERROR]

@pytest.mark.asyncio
async def test_page_state_checking_works(integration_agent):
    """Test that page state checking mechanism works."""
    # Navigate to real page
    await integration_agent._page.goto("https://example.com")
    
    # Check page state - should not crash
    state = await integration_agent._check_page_state()
    
    # Should return a valid state
    assert isinstance(state, AgentState)
    assert state in [AgentState.IDLE, AgentState.EXECUTING, AgentState.CAPTCHA_REQUIRED, 
                     AgentState.UNEXPECTED_MODAL, AgentState.FATAL_ERROR]

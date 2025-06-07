import pytest
import asyncio
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest_asyncio

from playwright.async_api import Page, BrowserContext, TimeoutError as PlaywrightTimeoutError, ElementHandle
from web_automation.core.browser_agent import PlaywrightBrowserAgent, ActionType
from web_automation.models.instructions import ClickInstruction, InstructionSet
from pydantic import ValidationError

# Fixture for mocking Playwright page
@pytest_asyncio.fixture
async def mock_page():
    page = AsyncMock(spec=Page)
    page.url = "http://example.com"
    page.title = AsyncMock(return_value="Example Domain")
    page.content = AsyncMock(return_value="<html><body>Example Content</body></html>")
    page.wait_for_selector = AsyncMock()
    page.keyboard = AsyncMock()
    page.mouse = AsyncMock()
    page.click = AsyncMock()
    page.fill = AsyncMock()
    page.goto = AsyncMock()
    return page

# Fixture for mocking Playwright browser context
@pytest_asyncio.fixture
async def mock_context():
    context = AsyncMock(spec=BrowserContext)
    return context

# Fixture for creating a PlaywrightBrowserAgent with mocked dependencies
@pytest_asyncio.fixture
async def agent(mock_page, mock_context):
    from web_automation.core.dependencies import BrowserAgentDependencies
    
    # Create proper dependencies
    dependencies = BrowserAgentDependencies(
        memory_manager=None,
        config={
            'identity_id': 'test_agent_retry',
            'headless': True,
            'browser_type': 'chromium'
        }
    )
    
    # Create agent with dependencies
    agent_instance = PlaywrightBrowserAgent(dependencies=dependencies)
    
    # Mock the browser infrastructure
    agent_instance._page = mock_page
    agent_instance._context = mock_context
    agent_instance._browser = AsyncMock()
    agent_instance._playwright = AsyncMock()
    
    # Mock state management (required for current implementation)
    agent_instance.current_state = "EXECUTING"  # Set state directly
    agent_instance._check_page_state = AsyncMock(return_value="EXECUTING")
    agent_instance._handle_state_transition = AsyncMock(return_value=True)
    
    return agent_instance

# Tests for retry logic in PlaywrightBrowserAgent
class TestPlaywrightBrowserAgentRetryLogic:
    @pytest.mark.asyncio
    async def test_instruction_succeeds_on_first_attempt(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction succeeds on the first try without retries."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", retry_attempts=3, retry_delay=100)
        
        # Mock the executor instead of _handle_click
        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        mock_page.click = AsyncMock()  # Mock the actual click method
        
        # Execute using the real method
        await agent._execute_instruction_with_memory(instruction, agent.identity_id)
        
        # Verify the page.click was called (this is what actually happens)
        mock_page.click.assert_called_once_with("#button")

    @pytest.mark.asyncio
    async def test_instruction_succeeds_after_retries(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction succeeds after a few failed attempts."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", retry_attempts=3, retry_delay=50)

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element

        # Mock click to fail twice then succeed
        mock_page.click = AsyncMock(side_effect=[
            PlaywrightTimeoutError("Attempt 1 failed"),
            PlaywrightTimeoutError("Attempt 2 failed"),
            None  # Success on 3rd attempt
        ])

        # Disable human-like delays for cleaner testing
        agent._human_like_delay = AsyncMock()

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Should succeed after retries
            await agent._execute_instruction_with_memory(instruction, agent.identity_id)

        # Verify retries occurred
        assert mock_page.click.call_count == 3
        
        # Now the counts should match exactly
        assert mock_sleep.call_count == 2  # Just the retry delays
        
        # Verify the delay values
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 0.05  # 50ms = 0.05 seconds

    @pytest.mark.asyncio
    async def test_instruction_fails_after_all_retries(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction fails if all retry attempts are exhausted."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", retry_attempts=2, retry_delay=50)

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element

        # Mock click to always fail
        mock_page.click = AsyncMock(side_effect=PlaywrightTimeoutError("Always fails"))

        # Disable human-like delays for cleaner testing
        agent._human_like_delay = AsyncMock()

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Should raise exception after all retries
            with pytest.raises(PlaywrightTimeoutError):
                await agent._execute_instruction_with_memory(instruction, agent.identity_id)

        # Verify all attempts were made
        assert mock_page.click.call_count == 2
        
        # Now the counts should match exactly
        assert mock_sleep.call_count == 1  # Just the retry delay
        
        # Verify the delay value
        for call in mock_sleep.call_args_list:
            assert call[0][0] == 0.05  # 50ms = 0.05 seconds

    @pytest.mark.asyncio 
    async def test_instruction_uses_default_retry_parameters(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that default retry_attempts=3 is used if not specified."""
        # Instruction without explicit retry_attempts
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button")
        
        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        mock_page.click = AsyncMock(side_effect=PlaywrightTimeoutError("Fails"))
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(PlaywrightTimeoutError):
                await agent._execute_instruction_with_memory(instruction, agent.identity_id)
        
        # Check default behavior (should use retry_attempts=3 from instruction default)
        assert mock_page.click.call_count == instruction.retry_attempts

    @pytest.mark.asyncio
    async def test_retry_logic_works_with_valid_parameters(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that retry logic works correctly with various valid parameters."""
        
        test_cases = [
            (1, 100),  # 1 attempt, 100ms delay
            (2, 500),  # 2 attempts, 500ms delay  
            (3, 1000), # 3 attempts, 1000ms delay
        ]
        
        for attempts, delay in test_cases:
            mock_page.reset_mock()
            
            instruction = ClickInstruction(
                type=ActionType.CLICK,
                selector="#button",
                retry_attempts=attempts,
                retry_delay=delay
            )
            
            mock_element = AsyncMock(spec=ElementHandle)
            mock_page.wait_for_selector.return_value = mock_element
            mock_page.click = AsyncMock(side_effect=PlaywrightTimeoutError("Fails"))
            
            # Disable human-like delays for cleaner testing
            agent._human_like_delay = AsyncMock()
            
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(PlaywrightTimeoutError):
                    await agent._execute_instruction_with_memory(instruction, agent.identity_id)
            
            # Verify correct number of attempts
            assert mock_page.click.call_count == attempts
            
            # Verify correct delay between attempts
            if attempts > 1:
                assert mock_sleep.call_count == attempts - 1
                for call in mock_sleep.call_args_list:
                    assert call[0][0] == delay / 1000.0  # asyncio.sleep takes seconds
            else:
                assert mock_sleep.call_count == 0  # No delay for single attempt

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_value, field_name", [
        ("abc", "retry_attempts"),
        ("xyz", "retry_delay"),
    ])
    async def test_pydantic_rejects_string_values(self, invalid_value, field_name):
        """Test that Pydantic properly rejects string values for numeric fields."""
        with pytest.raises(ValidationError) as exc_info:
            if field_name == "retry_attempts":
                ClickInstruction(
                    type=ActionType.CLICK,
                    selector="#button", 
                    retry_attempts=invalid_value,
                    retry_delay=1000
                )
            else:  # retry_delay
                ClickInstruction(
                    type=ActionType.CLICK,
                    selector="#button",
                    retry_attempts=3,
                    retry_delay=invalid_value
                )
        
        # Verify the validation error mentions the correct field and type
        error_str = str(exc_info.value)
        assert field_name in error_str
        assert "int_parsing" in error_str or "invalid" in error_str.lower()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_attempts, invalid_delay, expected_log_attempts, expected_log_delay", [
        (0, 500, "Invalid retry_attempts (0) for ActionType.CLICK. Defaulting to 1.", None),
        (-1, 500, "Invalid retry_attempts (-1) for ActionType.CLICK. Defaulting to 1.", None),
        (2, -100, None, "Invalid retry_delay_ms (-100) for ActionType.CLICK. Defaulting to 1000ms."),
        (0, -100, "Invalid retry_attempts (0) for ActionType.CLICK. Defaulting to 1.", "Invalid retry_delay_ms (-100) for ActionType.CLICK. Defaulting to 1000ms."),
    ])
    async def test_invalid_retry_parameters_are_defaulted(self, agent: PlaywrightBrowserAgent, mock_page,
                                                        invalid_attempts, invalid_delay, 
                                                        expected_log_attempts, expected_log_delay, caplog):
        """Test that invalid retry_attempts and retry_delay are defaulted and logged."""
        
        # Create instruction - only test numeric edge cases that pass Pydantic validation
        instruction = ClickInstruction(
            type=ActionType.CLICK, 
            selector="#button",
            retry_attempts=invalid_attempts, 
            retry_delay=invalid_delay
        )

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        
        # Mock the agent methods to fail and test retry logic
        mock_page.click = AsyncMock(side_effect=PlaywrightTimeoutError("Fails to trigger defaults"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(PlaywrightTimeoutError):
                await agent._execute_instruction_with_memory(instruction, agent.identity_id)
    
        # Check that appropriate warnings were logged
        if expected_log_attempts:
            assert expected_log_attempts in caplog.text
        if expected_log_delay:
            assert expected_log_delay in caplog.text

import pytest
import asyncio
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch, call

from web_automation.core.browser_agent import PlaywrightBrowserAgent, InstructionExecutionError
from web_automation.models.instructions import (InstructionSet, ClickInstruction, ActionType, 
                                                       NavigateInstruction, WaitInstruction, 
                                                       TypeInstruction, ScrollInstruction, ScreenshotInstruction, 
                                                       ExtractInstruction, EvaluateInstruction, UploadInstruction, 
                                                       DownloadInstruction, WaitCondition)
from web_automation.config.config_models import AntiDetectionConfig, Mem0AdapterConfig, BrowserConfig
from playwright.async_api import Page, BrowserContext, ElementHandle, PlaywrightTimeoutError


@pytest.fixture
async def mock_page():
    page = AsyncMock(spec=Page)
    page.url = "http://example.com"
    page.title = AsyncMock(return_value="Example Domain")
    page.content = AsyncMock(return_value="<html><body>Example Content</body></html>")
    page.wait_for_selector = AsyncMock(spec=ElementHandle)
    page.keyboard = AsyncMock()
    page.mouse = AsyncMock()
    return page

@pytest.fixture
async def mock_context():
    context = AsyncMock(spec=BrowserContext)
    return context

@pytest.fixture
async def agent(mock_page, mock_context):
    # Mock BrowserConfig and AntiDetectionConfig if needed, or use defaults
    browser_manager_config = BrowserConfig()
    anti_detection_config = AntiDetectionConfig()
    
    # For these tests, we don't need a real Mem0, so mock it or pass a minimal config
    mem0_config = Mem0AdapterConfig(enabled=False) 

    agent_instance = PlaywrightBrowserAgent(
        browser_manager_config=browser_manager_config,
        anti_detection_config=anti_detection_config,
        mem0_config=mem0_config,
        headless=True,
        user_agent=None,
        browser_args=None,
        viewport=None,
        default_timeout=5000,
        navigation_timeout=10000,
        js_injection_path=None,
        identity_id="test_agent_retry"
    )
    agent_instance._page = mock_page
    agent_instance._context = mock_context
    agent_instance.memory_manager = AsyncMock() # Mock memory manager
    agent_instance.captcha_solver = AsyncMock()
    agent_instance.is_initialized = True # Assume browser is initialized
    return agent_instance


class TestPlaywrightBrowserAgentRetryLogic:
    @pytest.mark.asyncio
    async def test_instruction_succeeds_on_first_attempt(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction succeeds on the first try without retries."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", retry_attempts=3, retry_delay=100)
        instruction_set = InstructionSet(instructions=[instruction])

        # Mock the handler to succeed on the first call
        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        agent._handle_click = AsyncMock()

        result = await agent.execute_instructions(instruction_set)

        assert result["success"] is True
        assert len(result["errors"]) == 0
        assert result["actions_completed"] == 1
        agent._handle_click.assert_called_once_with(instruction)
        # Ensure memory storage for success was called
        agent._store_execution_success.assert_called_once()
        agent._handle_execution_failure.assert_not_called()

    @pytest.mark.asyncio
    async def test_instruction_succeeds_after_retries(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction succeeds after a few failed attempts."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", retry_attempts=3, retry_delay=50) # Short delay for test speed
        instruction_set = InstructionSet(instructions=[instruction])

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        
        # Mock the handler to fail twice then succeed
        agent._handle_click = AsyncMock(side_effect=[
            PlaywrightTimeoutError("Attempt 1 failed"), 
            PlaywrightTimeoutError("Attempt 2 failed"), 
            None # Success on 3rd attempt
        ])

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await agent.execute_instructions(instruction_set)

        assert result["success"] is True
        assert len(result["errors"]) == 0
        assert result["actions_completed"] == 1
        assert agent._handle_click.call_count == 3
        mock_sleep.assert_has_calls([
            call(0.05), # 50ms delay
            call(0.05)  # 50ms delay
        ])
        assert mock_sleep.call_count == 2
        agent._store_execution_success.assert_called_once()
        agent._handle_execution_failure.assert_not_called() # Should not be called if final attempt succeeds

    @pytest.mark.asyncio
    async def test_instruction_fails_after_all_retries(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that an instruction fails if all retry attempts are exhausted."""
        instruction = TypeInstruction(type=ActionType.TYPE, selector="#input", text="test", retry_attempts=2, retry_delay=50)
        instruction_set = InstructionSet(instructions=[instruction])

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        
        # Mock the handler to always fail
        agent._handle_type = AsyncMock(side_effect=PlaywrightTimeoutError("Always fails"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await agent.execute_instructions(instruction_set)

        assert result["success"] is False
        assert len(result["errors"]) == 1
        assert "PlaywrightTimeoutError" in result["errors"][0]["type"] # Check for specific error type
        assert result["errors"][0]["message"] == "Always fails"
        assert result["actions_completed"] == 0
        assert agent._handle_type.call_count == 2
        mock_sleep.assert_called_once_with(0.05) # Only one sleep after the first failure
        agent._store_execution_success.assert_not_called()
        agent._handle_execution_failure.assert_called_once() # Called after all retries fail

    @pytest.mark.asyncio
    async def test_instruction_uses_default_retry_parameters(self, agent: PlaywrightBrowserAgent, mock_page):
        """Test that default retry_attempts=1 is used if not specified."""
        # Instruction without retry_attempts or retry_delay
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button") 
        instruction_set = InstructionSet(instructions=[instruction])

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        agent._handle_click = AsyncMock(side_effect=PlaywrightTimeoutError("Fails once"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await agent.execute_instructions(instruction_set)

        assert result["success"] is False # Should fail as default is 1 attempt
        assert agent._handle_click.call_count == 1 # Default 1 attempt
        mock_sleep.assert_not_called() # No retries, so no sleep
        agent._handle_execution_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_attempts, invalid_delay, expected_log_attempts, expected_log_delay", [
        (0, 500, "Invalid retry_attempts (0) for ActionType.CLICK. Defaulting to 1.", None),
        (-1, 500, "Invalid retry_attempts (-1) for ActionType.CLICK. Defaulting to 1.", None),
        ("abc", 500, "Invalid retry_attempts (abc) for ActionType.CLICK. Defaulting to 1.", None),
        (2, -100, None, "Invalid retry_delay_ms (-100) for ActionType.CLICK. Defaulting to 1000ms."),
        (2, "xyz", None, "Invalid retry_delay_ms (xyz) for ActionType.CLICK. Defaulting to 1000ms."),
        (0, -100, "Invalid retry_attempts (0) for ActionType.CLICK. Defaulting to 1.", "Invalid retry_delay_ms (-100) for ActionType.CLICK. Defaulting to 1000ms."),
    ])
    async def test_invalid_retry_parameters_are_defaulted(self, agent: PlaywrightBrowserAgent, mock_page, 
                                                        invalid_attempts, invalid_delay, 
                                                        expected_log_attempts, expected_log_delay, caplog):
        """Test that invalid retry_attempts and retry_delay are defaulted and logged."""
        instruction = ClickInstruction(type=ActionType.CLICK, selector="#button", 
                                       retry_attempts=invalid_attempts, retry_delay=invalid_delay)
        instruction_set = InstructionSet(instructions=[instruction])

        mock_element = AsyncMock(spec=ElementHandle)
        mock_page.wait_for_selector.return_value = mock_element
        # Make it fail to trigger retry logic path and logging
        agent._handle_click = AsyncMock(side_effect=PlaywrightTimeoutError("Fails to trigger defaults"))

        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await agent.execute_instructions(instruction_set)
        
        if expected_log_attempts:
            assert expected_log_attempts in caplog.text
        if expected_log_delay:
            assert expected_log_delay in caplog.text
        
        # It will attempt once with default if attempts were invalid, or 'invalid_attempts' times if only delay was invalid
        # If attempts were invalid (e.g., 0, -1, "abc"), it defaults to 1 attempt.
        # If attempts were valid (e.g., 2) but delay was invalid, it attempts 2 times.
        expected_call_count = invalid_attempts if isinstance(invalid_attempts, int) and invalid_attempts > 0 else 1
        assert agent._handle_click.call_count == expected_call_count

        # Check sleep calls based on actual attempts and defaulted delay
        if expected_call_count > 1:
             # Default delay is 1000ms if original was invalid
            actual_delay_for_sleep = invalid_delay if isinstance(invalid_delay, (int, float)) and invalid_delay >=0 else 1000
            assert mock_sleep.call_count == expected_call_count -1
            mock_sleep.assert_called_with(actual_delay_for_sleep / 1000.0)
        else:
            mock_sleep.assert_not_called()

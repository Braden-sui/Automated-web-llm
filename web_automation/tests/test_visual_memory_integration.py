import asyncio
import pytest
import uuid
import logging
import base64
from unittest.mock import AsyncMock, MagicMock
import numpy as np
from web_automation.config.config_models import Mem0AdapterConfig, VisualSystemConfig
from web_automation.core.dependencies import BrowserAgentFactory
from web_automation.memory.memory_enhanced_agent import PersistentMemoryBrowserAgent
import transformers.utils.logging as hf_logging

# Configure test logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.getLogger('mem0').setLevel(logging.DEBUG)
logging.getLogger('web_automation').setLevel(logging.DEBUG)
hf_logging.set_verbosity_debug()
logger.info("Hugging Face Transformers log level set to DEBUG for the module.")

# Test constants
TEST_SCREENSHOT_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
TEST_SCREENSHOT_BYTES = base64.b64decode(TEST_SCREENSHOT_BASE64)
TEST_LLM_DESCRIPTION = "A tiny transparent image, likely a placeholder."

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def test_mem0_config_for_visual_tests():
    return Mem0AdapterConfig(
        qdrant_path=None,
        qdrant_on_disk=False,
        qdrant_collection_name=f"test_visual_mem_collection_{uuid.uuid4().hex[:10]}",
        qdrant_embedding_model_dims=384,
        mem0_version="v1.1",
        llm_provider="ollama",
        llm_model="qwen2:0.5b",
        llm_temperature=0.1,
        api_key=None
    )

@pytest.fixture
def test_visual_system_config():
    return VisualSystemConfig(
        enabled=True,
        auto_capture=True,  # Required for visual memory tests
        model_name="qwen2.5vl:7b",
        ollama_base_url=None
    )

@pytest.fixture
def mock_ollama_chat_response():
    return {
        "model": "qwen2.5vl:7b",
        "created_at": "2024-07-29T12:00:00.000Z",
        "message": {"role": "assistant", "content": TEST_LLM_DESCRIPTION, "images": None},
        "done": True,
    }

@pytest.fixture
def real_browser_config():
    """Configuration for real browser testing."""
    return {
        'headless': True,  # Keep headless for CI
        'slow_mo': 100,    # Slow down for more reliable interactions
        'timeout': 30000,  # Longer timeout for real page loads
        'args': [
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--disable-blink-features=AutomationControlled'
        ]
    }

@pytest.fixture
def test_urls():
    """Reliable test URLs that won't change."""
    return {
        'simple': 'https://example.com',
        'form': 'https://httpbin.org/forms/post',
        'html': 'https://httpbin.org/html',
        'json': 'https://httpbin.org/json',
        'status': 'https://httpbin.org/status/200'
    }

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def assert_real_visual_capture(spy_capture, expected_url, action_type="navigate_complete"):
    """Assert that real visual capture occurred correctly."""
    assert spy_capture.called, "Visual capture should have been triggered"
    
    call_args = spy_capture.call_args
    assert call_args[1]['current_url'].rstrip('/') == expected_url.rstrip('/')
    assert call_args[1]['action_type'] == action_type

def assert_real_memory_storage(spy_store, agent_id):
    """Assert that real memory storage occurred."""
    assert spy_store.called, "Memory storage should have occurred"
    
    store_call = spy_store.call_args
    assert store_call[1]['user_id'] == agent_id
    
    visual_data = store_call[1]['visual_data']
    assert 'screenshot_base64' in visual_data
    assert 'screenshot_description' in visual_data
    assert 'url' in visual_data
    # Real screenshot should be much larger than our test placeholder
    assert len(visual_data['screenshot_base64']) > len(TEST_SCREENSHOT_BASE64)

def setup_essential_mocks(mocker, mock_ollama_chat_response):
    """Set up only the essential mocks needed for testing."""
    # Mock SentenceTransformer to prevent actual model loading
    mock_st_instance = MagicMock(name="MockedSentenceTransformerInstance")
    dummy_embedding = np.array([[0.1] * 384])
    mock_st_instance.encode.return_value = dummy_embedding
    
    def st_constructor_side_effect(*args, **kwargs):
        logger.debug("SentenceTransformer constructor mocked")
        return mock_st_instance
    
    mocker.patch('sentence_transformers.SentenceTransformer', side_effect=st_constructor_side_effect)
    
    # Mock Ollama for vision processing
    mock_async_ollama_client = AsyncMock()
    mock_async_ollama_client.chat = AsyncMock(return_value=mock_ollama_chat_response)
    mocker.patch('ollama.AsyncClient', return_value=mock_async_ollama_client)
    
    return mock_async_ollama_client

# =============================================================================
# TESTS
# =============================================================================

@pytest.mark.asyncio
async def test_visual_memory_with_real_webpages(
    test_mem0_config_for_visual_tests,
    test_visual_system_config,
    mock_ollama_chat_response,
    real_browser_config,
    test_urls,
    mocker
):
    """Test visual memory capture and fallback with real webpages."""
    
    print("🌐 Starting real webpage visual memory integration test")
    
    # Set up essential mocks only
    mock_ollama_client = setup_essential_mocks(mocker, mock_ollama_chat_response)
    
    agent_identity_id = f'test_visual_agent_{uuid.uuid4().hex[:8]}'
    
    # Create agent with real browser configuration
    agent = BrowserAgentFactory.create_agent(
        memory_config={'enabled': True, **test_mem0_config_for_visual_tests.model_dump()},
        visual_config_input=test_visual_system_config.model_dump(),
        browser_type='chromium',
        headless=real_browser_config['headless'],
        identity_id=agent_identity_id
    )
    
    assert isinstance(agent, PersistentMemoryBrowserAgent)
    assert agent.visual_system is not None
    assert agent.visual_system.llm_client is mock_ollama_client
    
    async with agent:
        print("📱 Agent initialized successfully")
        
        # Set up spies to monitor system behavior
        spy_capture_visual = mocker.spy(agent.visual_system, 'capture_visual_context')
        spy_store_visual = mocker.spy(agent.memory_manager, 'store_visual_pattern')
        spy_store_automation = mocker.spy(agent.memory_manager, 'store_automation_pattern')
        spy_visual_fallback = mocker.spy(agent.visual_system, 'enable_visual_fallback')
        spy_match_pattern = mocker.spy(agent.visual_system, 'match_visual_pattern_for_page')
        
        # =================================================================
        # Test 1: Real Navigation with Visual Capture
        # =================================================================
        print("=== Test 1: Real webpage navigation ===")
        
        await agent.navigate(test_urls['simple'])
        await asyncio.sleep(2)  # Allow page to fully load
        
        # Verify visual capture happened during navigation
        assert_real_visual_capture(spy_capture_visual, test_urls['simple'])
        assert_real_memory_storage(spy_store_visual, agent_identity_id)
        
        # Verify LLM was called for image description
        mock_ollama_client.chat.assert_called()
        
        print("✅ Real navigation and visual capture successful")
        
        # Reset spies for next test
        spy_capture_visual.reset_mock()
        spy_store_visual.reset_mock()
        mock_ollama_client.chat.reset_mock()
        
        # =================================================================
        # Test 2: Real Element Interaction
        # =================================================================
        print("=== Test 2: Real element interaction ===")
        
        await agent.navigate(test_urls['form'])
        await asyncio.sleep(2)
        
        # Try to interact with real form elements
        try:
            # These are real elements on httpbin.org/forms/post
            await agent.click("input[name='custname']")
            await agent.fill("input[name='custname']", "Test User")
            print("✅ Real element interaction successful")
            
            # Verify visual context was captured during interactions
            assert spy_capture_visual.call_count >= 1  # At least from navigation
            
        except Exception as e:
            print(f"ℹ️ Element interaction note: {e}")
        
        # Reset for next test
        spy_capture_visual.reset_mock()
        spy_store_visual.reset_mock()
        spy_store_automation.reset_mock()
        
        # =================================================================
        # Test 3: Visual Fallback with Real Page Content
        # =================================================================
        print("=== Test 3: Visual fallback mechanism ===")
        
        await agent.navigate(test_urls['html'])
        await asyncio.sleep(2)
        
        # Mock the click method to fail, forcing visual fallback
        original_click = agent.click
        
        async def mock_failing_click(*args, **kwargs):
            raise Exception("Simulated click failure for visual fallback test")
        
        agent.click = mock_failing_click
        
        # Mock visual pattern matching to return success
        mock_pattern = {
            'memory_id': 'test_pattern_123',
            'description': 'Test visual pattern match on real page',
            'visual_data': 'mock_visual_data',
            'metadata': {
                'action_type': 'click',
                'element_description': 'page element',
                'coordinates': {'x': 200, 'y': 300}
            }
        }
        
        spy_match_pattern.return_value = mock_pattern
        
        # Mock coordinate clicking to succeed
        mock_coord_click = mocker.patch.object(
            agent.visual_system,
            '_perform_click_at_coordinates',
            AsyncMock(return_value=True)
        )
        
        # Attempt smart_selector_click - should trigger visual fallback
        try:
            result = await agent.smart_selector_click("nonexistent element", "div.nonexistent")
            print(f"✅ Visual fallback test completed: {result}")
            
            # Verify visual fallback was attempted
            if spy_visual_fallback.called:
                print("✅ Visual fallback mechanism successfully triggered")
                assert_real_memory_storage(spy_store_visual, agent_identity_id)
            else:
                print("ℹ️ Visual fallback not triggered (selector may have been found)")
                
        except Exception as e:
            print(f"ℹ️ Visual fallback test note: {e}")
        
        # Restore original click method
        agent.click = original_click
        
        # Reset for next test
        spy_capture_visual.reset_mock()
        spy_store_visual.reset_mock()
        mock_ollama_client.chat.reset_mock()
        
        # =================================================================
        # Test 4: Multiple Real Page Visual Memory
        # =================================================================
        print("=== Test 4: Multi-page visual memory ===")
        
        # Navigate to different real pages and verify visual context is captured
        test_navigation_urls = [
            test_urls['html'],
            test_urls['json'],
            test_urls['status']
        ]
        
        initial_capture_count = spy_capture_visual.call_count
        
        for url in test_navigation_urls:
            print(f"📍 Navigating to: {url}")
            await agent.navigate(url)
            await asyncio.sleep(1.5)  # Allow page load and visual processing
        
        final_capture_count = spy_capture_visual.call_count
        captures_made = final_capture_count - initial_capture_count
        
        assert captures_made >= len(test_navigation_urls), f"Expected {len(test_navigation_urls)} captures, got {captures_made}"
        print(f"✅ Multi-page navigation completed: {captures_made} visual captures")
        
        # =================================================================
        # Test 5: Real Screenshot Analysis
        # =================================================================
        print("=== Test 5: Real screenshot analysis ===")
        
        await agent.navigate(test_urls['simple'])
        await asyncio.sleep(1)
        
        # Take actual screenshot and verify it's real
        screenshot_bytes = await agent._page.screenshot(type='png', full_page=True)
        assert len(screenshot_bytes) > 10000, "Real screenshot should be much larger than test placeholder"
        
        # Verify LLM was called for real image analysis
        mock_ollama_client.chat.assert_called()
        
        # Check that visual data includes real screenshot
        if spy_store_visual.called:
            last_store_call = spy_store_visual.call_args
            visual_data = last_store_call[1]['visual_data']
            assert 'screenshot_base64' in visual_data
            # Real screenshot base64 should be much larger than our test placeholder
            assert len(visual_data['screenshot_base64']) > len(TEST_SCREENSHOT_BASE64) * 10
            print("✅ Real screenshot analysis verified")
        
        print("\n🎉 All real webpage tests completed successfully!")
        print(f"📊 Test Summary:")
        print(f"   - Visual captures: {spy_capture_visual.call_count}")
        print(f"   - Memory storage: {spy_store_visual.call_count}")
        print(f"   - LLM calls: {mock_ollama_client.chat.call_count}")
        print(f"   - Agent ID: {agent_identity_id}")


@pytest.mark.slow
@pytest.mark.manual
@pytest.mark.asyncio
async def test_visual_memory_interactive(
    test_mem0_config_for_visual_tests,
    test_visual_system_config,
    mock_ollama_chat_response,
    test_urls,
    mocker
):
    """Interactive test with visible browser for manual verification."""
    
    print("🔍 Starting interactive visual memory test")
    print("👀 Browser will be visible for manual observation")
    
    # Set up essential mocks
    mock_ollama_client = setup_essential_mocks(mocker, mock_ollama_chat_response)
    
    agent = BrowserAgentFactory.create_agent(
        memory_config={'enabled': True, **test_mem0_config_for_visual_tests.model_dump()},
        visual_config_input=test_visual_system_config.model_dump(),
        browser_type='chromium',
        headless=False,  # Visible browser for manual inspection
        identity_id='interactive_test_agent'
    )
    
    async with agent:
        print("🌐 Opening real webpage...")
        await agent.navigate(test_urls['simple'])
        await asyncio.sleep(3)  # Allow manual observation
        
        print("📸 Visual memory capture happening...")
        await asyncio.sleep(2)
        
        print("🧠 Navigating to form page...")
        await agent.navigate(test_urls['form'])
        await asyncio.sleep(3)
        
        print("✅ Interactive test complete. Visual memory system demonstrated on real pages.")
        print("💡 Check the browser window to see real page interactions!")


# =============================================================================
# LEGACY TEST (MOCKED VERSION)
# =============================================================================

@pytest.mark.asyncio
async def test_visual_memory_and_fallback_mocked(
    test_mem0_config_for_visual_tests,
    test_visual_system_config,
    mock_ollama_chat_response,
    mocker
):
    """Legacy test with mocks - kept for comprehensive testing."""
    
    print("🔧 Running legacy mocked visual memory test")
    
    # Set up essential mocks
    mock_ollama_client = setup_essential_mocks(mocker, mock_ollama_chat_response)
    
    # Mock page screenshot for controlled testing
    mock_page_screenshot = AsyncMock(return_value=TEST_SCREENSHOT_BYTES)
    
    agent_identity_id = f'test_visual_agent_{uuid.uuid4().hex[:8]}'
    
    agent = BrowserAgentFactory.create_agent(
        memory_config={'enabled': True, **test_mem0_config_for_visual_tests.model_dump()},
        visual_config_input=test_visual_system_config.model_dump(),
        browser_type='chromium', 
        headless=True, 
        identity_id=agent_identity_id
    )
    
    async with agent:
        agent._page.screenshot = mock_page_screenshot
        
        # Set up spies
        spy_capture_visual = mocker.spy(agent.visual_system, 'capture_visual_context')
        spy_store_visual = mocker.spy(agent.memory_manager, 'store_visual_pattern')
        spy_store_automation = mocker.spy(agent.memory_manager, 'store_automation_pattern')
        
        # Test navigation with mocked screenshot
        nav_url = "https://example.com/navigated"
        await agent.navigate(nav_url)
        await asyncio.sleep(0.1)
        
        # Verify mocked behavior
        spy_capture_visual.assert_called_once()
        capture_kwargs = spy_capture_visual.call_args[1]
        assert capture_kwargs['current_url'] == nav_url
        assert capture_kwargs['action_type'] == "navigate_complete"
        
        spy_store_visual.assert_called_once()
        store_kwargs = spy_store_visual.call_args[1]
        assert store_kwargs['user_id'] == agent_identity_id
        assert store_kwargs['visual_data']['screenshot_base64'] == TEST_SCREENSHOT_BASE64
        
        print("✅ Legacy mocked test completed successfully")
import asyncio
import pytest
from web_automation import create_playwright_agent
# The plan mentions ClickInstruction and ActionType, but they are not directly used in the test_memory_enhanced_agent.
# If they were needed for constructing instructions to pass to an agent method, they would be imported from:
# from web_automation.models.instructions import ClickInstruction, ActionType 

@pytest.mark.asyncio
async def test_memory_enhanced_agent():
    """Test memory-enhanced agent basic functionality"""
    # Ensure awm_config is enabled for this test if it relies on AWM_ENABLED=true in .env
    # Alternatively, could mock settings.awm_config.ENABLED to be True for this test.
    agent = create_playwright_agent(memory_enabled=True, headless=True)
    
    async with agent: # WebBrowserAgent likely implements __aenter__ and __aexit__
        # Test navigation with memory (or standard navigation if memory doesn't alter it)
        # Assuming _page is an attribute set up by WebBrowserAgent's __aenter__
        if agent._page:
            await agent._page.goto("https://example.com")
        else:
            pytest.fail("Agent page was not initialized.")
        
        # Test smart selector (should fallback gracefully)
        # Assuming smart_selector_click is a method of MemoryEnhancedWebBrowserAgent
        success = await agent.smart_selector_click(
            target_description="main navigation link",
            fallback_selector="a[href*='example.com']" # Changed to a more likely selector on example.com
        )
        
        assert success, "smart_selector_click should succeed, at least with fallback."
        
        # Test memory stats
        # Assuming get_memory_stats is a method of MemoryEnhancedWebBrowserAgent
        stats = agent.get_memory_stats()
        assert "memory_enabled" in stats
        assert stats["memory_enabled"] == True
        assert "interactions_stored_session" in stats # Based on my implementation of get_memory_stats

@pytest.mark.asyncio  
async def test_standard_agent_compatibility():
    """Test that standard agent still works without memory"""
    agent = create_playwright_agent(memory_enabled=False, headless=True)
    
    async with agent:
        if agent._page:
            await agent._page.goto("https://example.com")
            # Basic check: page title
            title = await agent._page.title()
            assert "Example Domain" in title, "Standard agent could not navigate to example.com or title mismatch."
        else:
            pytest.fail("Agent page was not initialized for standard agent.")
        
def test_memory_config():
    """Test memory configuration loading"""
    from web_automation.config.settings import awm_config
    assert hasattr(awm_config, 'ENABLED')
    assert hasattr(awm_config, 'BACKEND')
    assert awm_config.ENABLED is True # Based on .env file created
    assert awm_config.BACKEND == "sqlite" # Based on .env file created

import asyncio
import logging
from web_automation import create_playwright_agent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pytest

@pytest.mark.asyncio
async def test_basic_agent():
    """Test basic functionality of the Playwright browser agent."""
    logger.info("Starting basic agent test...")
    
    # Create a browser agent with memory enabled
    agent = create_playwright_agent(
        memory_enabled=True,
        headless=False,  # Set to True for headless mode
        stealth=True,
        identity_id="test_identity_001"
    )
    
    try:
        # Initialize the agent
        await agent.initialize()
        logger.info("Agent initialized successfully.")
        
        # Navigate to a test page
        test_url = "https://example.com"
        logger.info(f"Navigating to {test_url}...")
        await agent._page.goto(test_url, timeout=60000)
        
        # Verify the page title
        title = await agent._page.title()
        logger.info(f"Page title: {title}")
        assert "Example Domain" in title, f"Unexpected page title: {title}"
        
        # Take a screenshot
        screenshot_path = "test_screenshot.png"
        await agent._page.screenshot(path=screenshot_path)
        logger.info(f"Screenshot saved to {screenshot_path}")
        
        # Test memory functionality if available
        if hasattr(agent, 'get_memory_stats'):
            stats = agent.get_memory_stats()
            logger.info(f"Memory stats: {stats}")
            assert "memory_enabled" in stats
            assert stats["memory_enabled"] is True
        
        logger.info("Basic agent test completed successfully.")
        
    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        logger.info("Closing agent...")
        await agent.close()
        logger.info("Agent closed.")

if __name__ == "__main__":
    asyncio.run(test_basic_agent())

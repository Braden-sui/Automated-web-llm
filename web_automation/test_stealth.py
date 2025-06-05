import asyncio
import argparse
import logging
from pathlib import Path

from .core.browser_agent import PlaywrightBrowserAgent
from .config.settings import general_config

# Setup basic logging for the test script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def main(identity_id: str):
    logger.info(f"--- Starting Stealth and Profile Test for Identity ID: {identity_id} ---")

    # Ensure necessary config directories exist (agent does this, but good for standalone test context)
    Path(general_config.DOWNLOADS_DIR).mkdir(parents=True, exist_ok=True)
    Path(general_config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    Path(general_config.SCREENSHOTS_DIR).mkdir(parents=True, exist_ok=True)
    # The profiles directory is created by the agent itself, relative to its own location.
    # No need to explicitly create it here if using the agent's default behavior.
    # Path(browser_config.PROFILES_DIR_NAME).mkdir(parents=True, exist_ok=True)

    agent = PlaywrightBrowserAgent(
        identity_id=identity_id,
        headless=False,  # Run in headed mode for visual inspection
        stealth=True
    )

    try:
        await agent.initialize()
        logger.info(f"PlaywrightBrowserAgent initialized for identity: {identity_id}")
        logger.info(f"Fingerprint Profile Used: {agent._fingerprint_profile}")

        # Navigate to a fingerprint testing site
        # test_url = "https://amiunique.org/fp"
        test_url = "https://fingerprint.com/products/fingerprintjs/"
        logger.info(f"Navigating to {test_url} for fingerprint inspection...")
        await agent._page.goto(test_url, timeout=60000) # Use agent's page directly for simplicity here
        
        logger.info(f"Navigation complete. Browser will remain open for 30 seconds for manual inspection.")
        logger.info(f"Please check the browser fingerprint details on the page.")
        await asyncio.sleep(30) # Keep browser open

    except Exception as e:
        logger.error(f"An error occurred during the test: {e}", exc_info=True)
    finally:
        logger.info("Closing browser agent...")
        await agent.close()
        logger.info(f"--- Test Finished for Identity ID: {identity_id} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PlaywrightBrowserAgent stealth and persistent profiles.")
    parser.add_argument("--identity-id", type=str, required=True, help="The AI digital identity ID to use for the test.")
    args = parser.parse_args()

    asyncio.run(main(args.identity_id))

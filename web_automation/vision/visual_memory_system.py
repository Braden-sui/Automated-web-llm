
import asyncio
from playwright.async_api import Page
from typing import Dict, Any, List, Optional
import logging
import base64
import ollama # Assuming ollama client library is used
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class VisualMemorySystem:
    """
    Leverages vision capabilities to create visual-based automation memory.
    """

    def __init__(self, llm_client: ollama.AsyncClient, memory_manager: Any, llm_model_name: str = os.getenv("VISUAL_SYSTEM_MODEL", "qwen2.5vl:7b")):
        """
        Initialize the VisualMemorySystem.

        Args:
            llm_client: An initialized Ollama async client for interacting with the vision-capable LLM.
            memory_manager: The system's memory manager instance for storing and retrieving visual patterns.
            llm_model_name: The name of the Ollama vision model to use.
        """
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.llm_model_name = llm_model_name
        logger.info(f"VisualMemorySystem initialized with LLM model: {self.llm_model_name}.")

    async def _describe_image_with_llm(self, image_bytes: bytes, prompt: str) -> Optional[str]:
        """Helper to get image description from LLM."""
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            response = await self.llm_client.chat(
                model=self.llm_model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64_image]
                    }
                ]
            )
            description = response.get('message', {}).get('content')
            if description:
                logger.info(f"LLM image description received: {description[:100]}...")
                return description.strip()
            else:
                logger.warning("LLM did not return a description content.")
                return None
        except Exception as e:
            logger.error(f"Error describing image with LLM: {e}")
            return None

    async def capture_visual_context(
        self, page: Page, user_id: str, action_type: str, target_element_selector: Optional[str] = None, current_url: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Captures visual context, describes it using the LLM, and stores it.

        Args:
            page: The Playwright page object.
            user_id: The ID of the user or agent for memory association.
            action_type: Description of the action being performed.
            target_element_selector: Optional selector of the element involved.
            current_url: The URL of the page, passed in case page.url is not yet updated.

        Returns:
            A dictionary containing visual descriptions and metadata, or None on failure.
        """
        logger.info(f"Capturing visual context for user '{user_id}', action: {action_type}, URL: {current_url or page.url}")
        
        try:
            screenshot_bytes = await page.screenshot(type='png', full_page=True)
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None
        
        prompt = f"Describe the visual layout and key elements on this webpage at URL {current_url or page.url}, focusing on the area related to '{action_type}'."
        if target_element_selector:
            prompt += f" Pay special attention to the element identified by selector '{target_element_selector}'."
        
        llm_text_description = await self._describe_image_with_llm(screenshot_bytes, prompt)
        
        if not llm_text_description:
            logger.warning("Could not generate LLM text description. Skipping visual context storage.")
            # Optionally, still store the image with minimal metadata if LLM fails
            # For now, we skip if LLM description is crucial.
            return None

        visual_context_metadata = {
            "action_type": action_type,
            "llm_text_description": llm_text_description,
            "target_element_selector": target_element_selector,
            "url": current_url or page.url,
            # Add other potential context here, e.g., extracted landmarks if we had them
        }
        
        if self.memory_manager:
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            prepared_visual_data = {
                "screenshot_base64": base64.b64encode(screenshot_bytes).decode('utf-8'),
                "screenshot_description": llm_text_description, # LLM's description of the visual
                "action_type": action_type,
                "url": current_url or page.url,
                "target_element_selector": target_element_selector,
                "visual_landmarks": visual_context_metadata.get("visual_landmarks", []),
                "layout_type": visual_context_metadata.get("layout_type", "unknown")
            }

            general_pattern_description = f"Visual pattern for {action_type} on {prepared_visual_data['url']}"
            
            # Metadata for store_visual_pattern can be for other non-core details.
            # Mem0BrowserAdapter's store_visual_pattern merges visual_data into its own metadata record.
            # Pass a minimal dict here if all relevant info is in prepared_visual_data.
            additional_metadata_for_call = {
                # Example: if there were other fields in visual_context_metadata not covered above
                # "original_raw_llm_output": visual_context_metadata.get("original_raw_llm_output")
            }

            self.memory_manager.store_visual_pattern(
                user_id=user_id,
                description=general_pattern_description,
                visual_data=prepared_visual_data,
                metadata=additional_metadata_for_call
            )
            logger.info(f"Visual context for '{action_type}' stored for user '{user_id}'.")
        else:
            logger.warning("Memory manager not available. Visual context not stored.")

        return visual_context_metadata

    async def match_visual_pattern_for_page(
        self, page: Page, user_id: str, target_action_description: str
    ) -> Optional[Dict[str, Any]]:
        """
        Takes a screenshot of the current page, gets its LLM description, and searches for the best
        matching stored visual pattern based on that description.

        Args:
            page: The current Playwright page object.
            user_id: The ID of the user or agent.
            target_action_description: Textual description of the target element or action.

        Returns:
            The best matching stored visual pattern (including its metadata and original visual_data),
            or None if no suitable match is found.
        """
        logger.info(f"Attempting to match visual pattern for page: '{target_action_description}' for user '{user_id}'.")
        try:
            current_screenshot_bytes = await page.screenshot(type='png', full_page=True)
        except Exception as e:
            logger.error(f"Failed to take screenshot for visual matching: {e}")
            return None

        prompt = f"Describe the current visual layout and key elements on this webpage, specifically focusing on elements related to '{target_action_description}'."
        current_llm_description = await self._describe_image_with_llm(current_screenshot_bytes, prompt)

        if not current_llm_description:
            logger.warning("Could not get LLM description for current page. Cannot match visual patterns.")
            return None

        if not self.memory_manager:
            logger.warning("Memory manager not available. Cannot search visual patterns.")
            return None

        # Search stored patterns using the new LLM description of the current page
        # Mem0BrowserAdapter.search_visual_patterns expects a query_description
        stored_patterns_raw = self.memory_manager.search_visual_patterns(
            query_description=current_llm_description, # Search based on current page's description
            user_id=user_id,
            limit=5 # Get a few candidates
        )

        if not stored_patterns_raw:
            logger.info("No visual patterns found in memory for the given query description.")
            return None

        # TODO: Implement more sophisticated similarity scoring if needed.
        # For now, assume the first result from semantic search is the best candidate if its description is relevant.
        # The 'stored_patterns_raw' from Mem0 search should ideally be ranked by relevance.
        best_match = stored_patterns_raw[0] # Assuming search_visual_patterns returns sorted results
        
        # Ensure the best_match has the necessary structure (it comes from Mem0BrowserAdapter.search_visual_patterns)
        # It should be a dict with 'memory' (original description), 'metadata' (original metadata), 'id'.
        # The original metadata should contain 'llm_text_description' and potentially 'coordinates' if stored previously.
        if best_match and best_match.get('metadata') and best_match['metadata'].get('llm_text_description'):
            logger.info(f"Best visual pattern match found: ID {best_match.get('id')}, Description: {best_match['metadata']['llm_text_description'][:100]}...")
            # The 'visual_data' (original screenshot) is also available in best_match['metadata']['screenshot_base64']
            return {
                'memory_id': best_match.get('id'),
                'description': best_match['metadata']['llm_text_description'], # LLM desc of stored image
                'visual_data': best_match['metadata'].get('screenshot_base64'), # Base64 of stored image
                'metadata': best_match['metadata'] # All other stored metadata
            }
        else:
            logger.warning("Best match from search did not have expected structure or LLM description.")
            return None

    async def _perform_click_at_coordinates(self, page: Page, coordinates: Dict, description: str) -> bool:
        """Placeholder: Performs a click at given coordinates. TODO: Implement robustly."""
        if not all(k in coordinates for k in ['x', 'y']):
            logger.error(f"Invalid coordinates for visual click: {coordinates}. Missing 'x' or 'y'.")
            return False
        try:
            # Ensure x and y are numbers
            x = float(coordinates['x'])
            y = float(coordinates['y'])
            logger.info(f"Attempting visual click at x={x}, y={y} for '{description}'.")
            # Playwright's click takes x, y relative to the top-left of the viewport.
            await page.mouse.click(x, y)
            logger.info(f"Successfully performed visual click at x={x}, y={y} for '{description}'.")
            return True
        except Exception as e:
            logger.error(f"Error performing visual click at {coordinates} for '{description}': {e}")
            return False

    async def enable_visual_fallback(
        self, page: Page, user_id: str, failed_action_description: str, failed_selector: Optional[str],
        current_screenshot_base64: Optional[str] # Screenshot of the page AT THE TIME of fallback
    ) -> bool:
        """
        Compares current visual state with stored patterns.

        Args:
            current_page: The Playwright page object of the current state.
            user_id: The ID of the user or agent.
            failed_action_description: Description of the action that failed.
            failed_selector: The selector that failed.
            current_screenshot_base64: Base64 encoded screenshot of the current page.

        Returns:
            True if action was successful using visual fallback, False otherwise.
        """
        logger.warning(
            f"Selector '{failed_selector}' failed for user '{user_id}', action '{failed_action_description}'. Attempting visual fallback."
        )
        
        # Use the new method to get a single best match based on current page's LLM description
        best_match_pattern = await self.match_visual_pattern_for_page(page, user_id, failed_action_description)

        store_desc_prefix = f"Visual fallback for '{failed_action_description}' (selector: {failed_selector or 'N/A'})"
        common_metadata = {
            "action_type": "visual_fallback_click",
            "original_target_description": failed_action_description,
            "original_failed_selector": failed_selector,
            "url": page.url
        }

        if best_match_pattern:
            coordinates = best_match_pattern.get('metadata', {}).get('coordinates')
            element_desc_from_pattern = best_match_pattern.get('description', 'unknown element')

            if not coordinates:
                logger.warning(f"Visual match found (ID: {best_match_pattern.get('memory_id')}), but no coordinates in its metadata to perform click.")
                if self.memory_manager and current_screenshot_base64:
                    store_desc_prefix = f"Visual fallback for '{failed_action_description}' (selector: {failed_selector or 'N/A'})"
                    self.memory_manager.store_visual_pattern(
                        user_id=user_id,
                        description=f"{store_desc_prefix}: Failed - Matched pattern (ID: {best_match_pattern.get('memory_id')}) lacks coordinate data.",
                        visual_data=current_screenshot_base64,
                        metadata={"action_type": "visual_fallback_click", "original_target_description": failed_action_description, "original_failed_selector": failed_selector, "url": page.url, "status": "failure_no_coordinates", "matched_pattern_id": best_match_pattern.get('memory_id')}
                    )
                return False

            # Attempt to perform the click using the coordinates
            click_success = await self._perform_click_at_coordinates(page, coordinates, f"visual fallback for {failed_action_description}")


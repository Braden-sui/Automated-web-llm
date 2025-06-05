# web_automation/captcha/detector.py
import asyncio
import logging
from typing import Optional, Dict, List, Any
import playwright.async_api as playwright_api
from PIL import Image
import io
import base64

# Assuming exceptions are in the same package
# from .exceptions import CaptchaNotFound 

logger = logging.getLogger(__name__)

class ImageCaptchaDetector:
    """
    Detects simple image-based CAPTCHAs on a webpage.
    Focuses on <img> tags and associated input fields.
    Skips complex CAPTCHAs like reCAPTCHA.
    """

    async def find_image_captcha(self, page: playwright_api.Page) -> Optional[Dict[str, playwright_api.ElementHandle]]:
        """
        Finds solvable image CAPTCHAs on the page.

        Returns:
            Optional[Dict[str, ElementHandle]]: A dictionary containing 'image_element'
                                                and 'input_element' if a solvable CAPTCHA is found,
                                                otherwise None.
        """
        logger.info("DETECTOR: Searching for CAPTCHAs...")

        # First, check for reCAPTCHA v2
        recaptcha_info = await self.find_recaptcha_v2(page)
        if recaptcha_info:
            return recaptcha_info

        logger.info("DETECTOR: No reCAPTCHA v2 found. Searching for generic image CAPTCHAs...")

        # Strategy 1: Look for <img> tags with 'captcha' in src or alt
        # More specific selectors can be added based on common patterns
        possible_captcha_images = await page.query_selector_all(
            'img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"], img[class*="captcha"]'
        )

        if not possible_captcha_images:
            logger.info("DETECTOR: No obvious CAPTCHA image tags found by common keywords.")
            # Strategy 2: Look for images near input fields that might be CAPTCHAs
            # This is more complex and will be refined later.
            # For now, let's try a simpler approach for images that are visible.
            all_images = await page.query_selector_all('img')
            visible_images = []
            for img in all_images:
                if await img.is_visible() and await img.get_attribute("src"):
                    # Basic check for image dimensions (not too small, not too large)
                    bounding_box = await img.bounding_box()
                    if bounding_box and 50 < bounding_box['width'] < 400 and 30 < bounding_box['height'] < 200:
                        visible_images.append(img)
            
            if not visible_images:
                logger.info("DETECTOR: No visible images found that meet basic dimension criteria.")
                return None
            possible_captcha_images.extend(visible_images)


        for img_element in possible_captcha_images:
            try:
                src = await img_element.get_attribute("src")
                alt = await img_element.get_attribute("alt")
                logger.debug(f"DETECTOR: Evaluating potential CAPTCHA image: src='{src}', alt='{alt}'")

                if not await img_element.is_visible():
                    logger.debug(f"DETECTOR: Image {src or alt} is not visible, skipping.")
                    continue

                # Attempt to find an associated input field
                # This logic is basic and needs to be made more robust.
                # It looks for an input field immediately following or preceding the image,
                # or within a small number of sibling elements, or within a shared parent.
                
                input_field = await self._find_associated_input_field(page, img_element)

                if input_field:
                    logger.info(f"DETECTOR: Found potential CAPTCHA image ('{src or alt}') with an associated input field.")
                    return {"image_element": img_element, "input_element": input_field}
                else:
                    logger.debug(f"DETECTOR: No associated input field found for image '{src or alt}'.")

            except Exception as e:
                logger.warning(f"DETECTOR: Error evaluating potential CAPTCHA image: {e}", exc_info=True)
                continue
        
        logger.info("DETECTOR: No solvable image CAPTCHA found after checking all candidates.")
        return None

    async def extract_recaptcha_challenge_images(self, challenge_frame: playwright_api.Frame) -> Optional[Dict[str, Any]]:
        """
        Extracts the instruction text and image grid from the reCAPTCHA v2 challenge frame.
        """
        try:
            # Get the instruction text
            instruction_element = await challenge_frame.query_selector(".rc-imageselect-instructions .rc-imageselect-desc")
            instruction_text = await instruction_element.inner_text() if instruction_element else ""
            logger.debug(f"RECATCHA: Challenge instruction: {instruction_text}")

            # Get the main challenge image (if any, sometimes it's just a grid)
            main_image_element = await challenge_frame.query_selector(".rc-image-tile-target img")
            main_image_bytes = None
            if main_image_element:
                main_image_bytes = await main_image_element.screenshot()
                logger.debug("RECATCHA: Main challenge image extracted.")

            # Get the grid images
            grid_image_elements = await challenge_frame.query_selector_all(".rc-imageselect-tile")
            grid_images_data = []
            for i, tile_element in enumerate(grid_image_elements):
                # Each tile might contain an img or just be a background image
                # We'll screenshot the tile element itself to capture the image
                tile_bytes = await tile_element.screenshot()
                if tile_bytes:
                    grid_images_data.append({"index": i, "bytes": tile_bytes, "element": tile_element})
                logger.debug(f"RECATCHA: Extracted grid image {i}.")

            if not grid_images_data and not main_image_bytes:
                logger.warning("RECATCHA: No challenge images found in reCAPTCHA frame.")
                return None

            return {
                "instruction": instruction_text,
                "main_image": main_image_bytes,
                "grid_images": grid_images_data
            }
        except Exception as e:
            logger.error(f"RECATCHA: Error extracting reCAPTCHA challenge images: {e}", exc_info=True)
            return None

    async def _find_associated_input_field(self, page: playwright_api.Page, image_element: playwright_api.ElementHandle) -> Optional[playwright_api.ElementHandle]:
        """
        Tries to find a text input field associated with the given CAPTCHA image element.
        This is a helper method and its logic can be quite heuristic.
        """
        # Try common patterns: input after image, input inside a shared parent, etc.
        # Pattern 1: Input immediately after the image or its parent
        xpath_strategies = [
            "./following-sibling::input[@type='text']",
            "./following-sibling::*[1]/input[@type='text']", # Input within next sibling
            "./parent::*/following-sibling::input[@type='text']",
            "./parent::*/descendant::input[@type='text']", # Most common: input within same container
            "//input[@type='text'][contains(@name, 'captcha') or contains(@id, 'captcha') or contains(@class, 'captcha')]" # General fallback
        ]
        
        for i, strategy in enumerate(xpath_strategies):
            try:
                if strategy.startswith("//"): # Global search
                    associated_input = await page.query_selector(strategy)
                else: # Relative search from img_element
                    associated_input = await img_element.query_selector(strategy)
                
                if associated_input and await associated_input.is_visible() and await associated_input.is_enabled():
                    logger.debug(f"DETECTOR: Found input field using XPath strategy #{i+1}")
                    return associated_input
            except Exception as e:
                logger.debug(f"DETECTOR: XPath strategy #{i+1} failed or no element found: {e}")
        
        # Fallback: Look for any visible text input field if specific association fails (less reliable)
        # This might be too broad, consider proximity if possible.
        # For now, if direct association fails, we assume no clear input field.
        logger.debug("DETECTOR: Could not find a clearly associated input field using common XPath strategies.")
        return None

    async def find_recaptcha_v2(self, page: playwright_api.Page) -> Optional[Dict[str, playwright_api.ElementHandle | playwright_api.Frame]]:
        """
        Detects reCAPTCHA v2 checkbox and challenge iframes.
        Returns a dictionary with 'checkbox_frame', 'checkbox_element', and optionally 'challenge_frame'.
        """
        logger.info("DETECTOR: Searching for reCAPTCHA v2 checkbox iframe...")
        try:
            # Find the main reCAPTCHA iframe (checkbox)
            checkbox_frame_element = await page.query_selector("iframe[src*='google.com/recaptcha/api2/anchor']")
            if not checkbox_frame_element:
                logger.debug("DETECTOR: reCAPTCHA v2 checkbox iframe not found.")
                return None

            checkbox_frame: playwright_api.Frame = await checkbox_frame_element.content_frame()
            if not checkbox_frame:
                logger.warning("DETECTOR: Could not get content frame for reCAPTCHA checkbox.")
                return None

            # Check if the checkbox is visible and enabled
            checkbox_element = await checkbox_frame.query_selector("#recaptcha-anchor")
            if not checkbox_element or not await checkbox_element.is_visible() or not await checkbox_element.is_enabled():
                logger.warning("DETECTOR: reCAPTCHA checkbox element not found or not interactable.")
                return None

            logger.info("DETECTOR: reCAPTCHA v2 checkbox found.")
            
            result = {
                "type": "recaptcha_v2",
                "checkbox_frame": checkbox_frame,
                "checkbox_element": checkbox_element
            }

            # Check for the challenge iframe (might not be immediately present)
            challenge_frame_element = await page.query_selector("iframe[src*='google.com/recaptcha/api2/bframe']")
            if challenge_frame_element:
                challenge_frame: playwright_api.Frame = await challenge_frame_element.content_frame()
                if challenge_frame:
                    logger.info("DETECTOR: reCAPTCHA v2 challenge frame found.")
                    result["challenge_frame"] = challenge_frame
                else:
                    logger.debug("DETECTOR: Could not get content frame for reCAPTCHA challenge.")
                    result["challenge_frame"] = None
            else:
                result["challenge_frame"] = None

            return result

        except Exception as e:
            logger.error(f"DETECTOR: Error detecting reCAPTCHA v2: {e}", exc_info=True)
            return None

    async def extract_captcha_image(self, page: playwright_api.Page, captcha_image_element: playwright_api.ElementHandle) -> Optional[bytes]:
        """
        Extracts the CAPTCHA image as bytes for vision processing.
        Also attempts to ensure the image is fully loaded.
        """
        try:
            # Ensure image is loaded (Playwright usually handles this, but an explicit check can be useful)
            # For dynamic images, waiting for a specific state or event might be needed.
            # Here, we rely on screenshotting the element which implies it's rendered.
            if not await captcha_image_element.is_visible():
                logger.warning("DETECTOR: CAPTCHA image element is not visible for extraction.")
                return None

            image_bytes = await captcha_image_element.screenshot()
            if not image_bytes:
                logger.warning("DETECTOR: Failed to screenshot CAPTCHA image element (empty bytes).")
                return None
            
            logger.info("DETECTOR: CAPTCHA image extracted successfully.")
            return image_bytes
        except Exception as e:
            logger.error(f"DETECTOR: Failed to extract CAPTCHA image: {e}", exc_info=True)
            return None

    async def get_input_field(self, page: playwright_api.Page, captcha_info: Dict[str, playwright_api.ElementHandle]) -> Optional[playwright_api.ElementHandle]:
        """
        Returns the text input field ElementHandle associated with the CAPTCHA.
        The captcha_info dictionary is expected to be the output of find_image_captcha.
        """
        input_element = captcha_info.get("input_element")
        if input_element and isinstance(input_element, playwright_api.ElementHandle):
            if await input_element.is_visible() and await input_element.is_enabled():
                return input_element
            else:
                logger.warning("DETECTOR: Associated input field from captcha_info is no longer visible or enabled.")
        logger.warning("DETECTOR: No valid input_element found in captcha_info or it's not usable.")
        return None


    async def submit_solution(self, page: playwright_api.Page, input_element: playwright_api.ElementHandle, solution: str) -> bool:
        """
        Types the solution into the input field and attempts to submit.
        Submission logic might need to be adapted based on common form patterns.
        """
        try:
            if not await input_element.is_visible() or not await input_element.is_enabled():
                logger.error("SUBMITTER: CAPTCHA input field is not visible or enabled for typing.")
                return False

            await input_element.fill(solution)
            logger.info(f"SUBMITTER: Typed solution '{solution}' into CAPTCHA input field.")

            # Attempt to submit. This is highly heuristic.
            # 1. Try pressing Enter on the input field itself.
            try:
                await input_element.press("Enter")
                logger.info("SUBMITTER: Pressed Enter on input field. Assuming submission.")
                # Add a small delay to allow page to react
                await page.wait_for_timeout(1000) 
                return True # Assume success for now, actual verification is harder
            except Exception as e:
                logger.warning(f"SUBMITTER: Pressing Enter on input field failed or not applicable: {e}")

            # 2. Look for a submit button near the input field or a common submit button.
            # This requires more advanced element finding logic (e.g., parent form's submit button)
            # For now, we'll keep it simple.
            # Example: submit_button = await page.query_selector("form input[type='submit'], form button[type='submit']")
            # if submit_button and await submit_button.is_visible():
            #    await submit_button.click()
            #    logger.info("SUBMITTER: Clicked a form submit button.")
            #    return True

            logger.info("SUBMITTER: Solution typed. No explicit submit action taken beyond pressing Enter. Manual verification may be needed.")
            return True # Returning true as typing was successful
        except Exception as e:
            logger.error(f"SUBMITTER: Failed to type or submit CAPTCHA solution: {e}", exc_info=True)
            return False

# web_automation/captcha/captcha_handler.py
import asyncio
import logging
from typing import Dict, Optional, Any # Changed List to Any for get_stats
import playwright.async_api as playwright_api

from .detector import ImageCaptchaDetector
from .vision_solver import VisionCaptchaSolver
from .exceptions import CaptchaError, CaptchaNotFound, CaptchaSolveFailed, VisionModelUnavailable
from ..config import settings # For SimpleCaptchaHandler defaults

logger = logging.getLogger(__name__)

class SimpleCaptchaHandler:
    """
    Main CAPTCHA handling orchestrator.
    Coordinates detection, solving, and submission of image CAPTCHAs.
    """
    
    def __init__(self, vision_model_name: Optional[str] = None, max_attempts_override: Optional[int] = None):
        self.detector = ImageCaptchaDetector()
        
        _captcha_config = settings.captcha_config
        
        _model_to_use = vision_model_name if vision_model_name is not None else _captcha_config.CAPTCHA_VISION_MODEL_NAME
        _attempts_to_use = max_attempts_override if max_attempts_override is not None else _captcha_config.CAPTCHA_MAX_ATTEMPTS
        
        self.solver = VisionCaptchaSolver(model_name=_model_to_use)
        self.max_attempts = _attempts_to_use
        
        logger.info(f"HANDLER: Initialized with vision model '{_model_to_use}' and max attempts '{_attempts_to_use}'.")
        
        self.total_handled = 0
        self.total_solved = 0
        
    async def handle_page_captchas(self, page: playwright_api.Page) -> bool:
        """
        Main method: detect and solve any CAPTCHAs on current page.
        
        Returns:
            bool: True if no CAPTCHAs found or all were solved successfully
        """
        logger.info("HANDLER: Starting CAPTCHA detection and solving process...")
        
        if not self.solver.is_model_available():
            logger.warning("HANDLER: No vision model available, skipping CAPTCHA solving")
            # Consider if this should return True or False based on requirements.
            # If no model means can't proceed, False might be more appropriate.
            return False # Changed to False as inability to solve is a failure condition
        
        try:
            captcha_info = await self.detector.find_image_captcha(page)
            
            if not captcha_info:
                logger.info("HANDLER: No solvable CAPTCHAs found on page")
                return True
            
            self.total_handled += 1 # Increment when a CAPTCHA is found and attempted
            
            success = False
            if captcha_info.get("type") == "recaptcha_v2":
                logger.info("HANDLER: Handling reCAPTCHA v2...")
                success = await self._handle_recaptcha_v2(page, captcha_info)
            else:
                logger.info("HANDLER: Handling generic image CAPTCHA...")
                success = await self.solve_single_captcha(page, captcha_info)
            
            if success:
                self.total_solved += 1
                logger.info("HANDLER: CAPTCHA solved successfully!")
                return True
            else:
                logger.warning("HANDLER: Failed to solve CAPTCHA")
                return False
                
        except CaptchaError as e:
            logger.error(f"HANDLER: CAPTCHA handling error: {e}")
            return False
        except Exception as e:
            logger.error(f"HANDLER: Unexpected error during CAPTCHA handling: {e}", exc_info=True)
            return False
    
    async def solve_single_captcha(self, page: playwright_api.Page, captcha_info: Dict) -> bool:
        image_element = captcha_info.get("image_element")
        input_element = captcha_info.get("input_element")
        
        if not image_element or not input_element:
            logger.error("HANDLER: Invalid captcha_info - missing image or input element")
            return False
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                logger.info(f"HANDLER: CAPTCHA solve attempt {attempt}/{self.max_attempts}")
                
                image_bytes = await self.detector.extract_captcha_image(page, image_element)
                if not image_bytes:
                    logger.error("HANDLER: Failed to extract CAPTCHA image for attempt {attempt}")
                    if attempt < self.max_attempts: await asyncio.sleep(1); continue
                    else: return False # Failed all attempts to extract
                
                # Analyze CAPTCHA type (could be done once if image doesn't change)
                captcha_type = await self.solver.analyze_captcha_type(image_bytes)
                logger.info(f"HANDLER: Attempt {attempt} - Detected CAPTCHA type: {captcha_type}")
                
                solution: Optional[str] = None
                if captcha_type == "math":
                    solution = await self.solver.solve_math_captcha(image_bytes)
                elif captcha_type == "text" or captcha_type == "unknown": # Try text solving for unknown too
                    solution = await self.solver.solve_text_captcha(image_bytes)
                else:
                    logger.warning(f"HANDLER: Attempt {attempt} - Unknown CAPTCHA type '{captcha_type}' not processed.")
                    if attempt < self.max_attempts: await asyncio.sleep(1); continue
                    else: return False

                if not solution:
                    logger.warning(f"HANDLER: Attempt {attempt} - No solution found by solver.")
                    if attempt < self.max_attempts: await asyncio.sleep(1); continue
                    else: return False
                
                logger.info(f"HANDLER: Attempt {attempt} - Solution candidate: '{solution}'")
                
                submission_successful = await self.detector.submit_solution(page, input_element, solution)
                
                if submission_successful:
                    await page.wait_for_timeout(settings.captcha_config.CAPTCHA_POST_SUBMIT_DELAY_MS) # Use config
                    
                    if await self._verify_captcha_solved(page, captcha_info, image_bytes):
                        logger.info(f"HANDLER: CAPTCHA verified as solved on attempt {attempt}")
                        return True
                    else:
                        logger.warning(f"HANDLER: Attempt {attempt} - Solution submitted but CAPTCHA not verified as solved.")
                else:
                    logger.warning(f"HANDLER: Attempt {attempt} - Failed to submit solution via detector.")
                
                if attempt < self.max_attempts:
                    logger.info(f"HANDLER: Waiting before retry attempt {attempt+1}...")
                    await asyncio.sleep(settings.captcha_config.CAPTCHA_RETRY_DELAY_MS / 1000.0) # Use config
                    # It's possible the image element became stale, re-find it if necessary or ensure detector handles it.
                    # For simplicity, we assume image_element is still valid or find_image_captcha would be called again.
                
            except CaptchaSolveFailed as e:
                logger.warning(f"HANDLER: Attempt {attempt} - Solving failed directly: {e}")
            except VisionModelUnavailable as e:
                logger.error(f"HANDLER: Vision model unavailable: {e}. Aborting CAPTCHA attempts.")
                return False # Cannot proceed without vision model
            except Exception as e:
                logger.error(f"HANDLER: Attempt {attempt} - Unexpected error in solve_single_captcha: {e}", exc_info=True)
            
            if attempt >= self.max_attempts: break # Exit loop if max attempts reached
        
        logger.error(f"HANDLER: Failed to solve CAPTCHA after {self.max_attempts} attempts")
        return False
    
    async def _verify_captcha_solved(self, page: playwright_api.Page, original_captcha_info: Dict, original_image_bytes: Optional[bytes]) -> bool:
        try:
            image_element = original_captcha_info.get("image_element")
            if image_element:
                try:
                    if not await image_element.is_visible(timeout=1000): # Quick check
                        logger.debug("VERIFICATION: Original CAPTCHA image no longer visible.")
                        return True
                    
                    # Check if image content changed significantly (more robust than src)
                    if original_image_bytes:
                        current_image_bytes = await image_element.screenshot()
                        if current_image_bytes != original_image_bytes:
                             # Could use an image diffing library for more accuracy
                            if len(current_image_bytes) != len(original_image_bytes): # Basic check
                                logger.debug("VERIFICATION: CAPTCHA image content/size appears to have changed (new CAPTCHA?).")
                                return False # Assume new CAPTCHA means failure of previous one
                except Exception:
                    logger.debug("VERIFICATION: Original CAPTCHA element likely detached or stale.")
                    return True # Element gone is often a sign of success
            
            # Check for common success/error text patterns on the page
            # This is highly site-specific and may need customization
            body_text = await page.locator('body').inner_text()
            if any(s.lower() in body_text.lower() for s in settings.captcha_config.CAPTCHA_SUCCESS_TEXTS):
                logger.debug("VERIFICATION: Found success text pattern on page.")
                return True
            if any(s.lower() in body_text.lower() for s in settings.captcha_config.CAPTCHA_FAILURE_TEXTS):
                logger.debug("VERIFICATION: Found failure text pattern on page.")
                return False

            # Fallback: if a new CAPTCHA is presented by the detector, the old one likely failed or page reloaded with new one
            current_captcha_info = await self.detector.find_image_captcha(page)
            if current_captcha_info and current_captcha_info.get("image_element") != image_element:
                logger.debug("VERIFICATION: A new, different CAPTCHA image was found.")
                return False # Old one failed or page reloaded
            if not current_captcha_info and not await image_element.is_visible(timeout=500):
                 logger.debug("VERIFICATION: No new CAPTCHA found and old one is gone.")
                 return True

            logger.debug("VERIFICATION: No clear success/failure indicators. Assuming previous action outcome dictates success.")
            # This verification is tricky. If no errors and no new CAPTCHA, assume it worked.
            # The calling code might need to check page state (e.g. URL change, expected element appearing).
            return True # Default to true if no obvious failure signs
            
        except Exception as e:
            logger.warning(f"VERIFICATION: Error during CAPTCHA verification: {e}", exc_info=True)
            return False

    async def _handle_recaptcha_v2(self, page: playwright_api.Page, recaptcha_info: Dict) -> bool:
        """
        Handles the reCAPTCHA v2 checkbox and challenge.
        """
        checkbox_frame: playwright_api.Frame = recaptcha_info["checkbox_frame"]
        checkbox_element: playwright_api.ElementHandle = recaptcha_info["checkbox_element"]
        challenge_frame: Optional[playwright_api.Frame] = recaptcha_info["challenge_frame"]

        if not checkbox_frame or not checkbox_element:
            logger.error("RECATCHA: Missing checkbox frame or element info.")
            return False

        try:
            logger.info("RECATCHA: Clicking reCAPTCHA v2 checkbox...")
            await checkbox_element.click()
            await page.wait_for_timeout(settings.captcha_config.CAPTCHA_POST_SUBMIT_DELAY_MS) # Wait for potential challenge

            # Check if challenge appeared or if it was instantly solved
            # Need to re-detect challenge frame as it might appear after click
            if not challenge_frame:
                # Re-query for challenge frame after clicking checkbox
                challenge_frame_element = await page.query_selector("iframe[src*='google.com/recaptcha/api2/bframe']")
                if challenge_frame_element:
                    challenge_frame = await challenge_frame_element.content_frame()

            if challenge_frame:
                logger.info("RECATCHA: reCAPTCHA v2 challenge appeared. Attempting to solve...")
                return await self._solve_recaptcha_v2_challenge(page, challenge_frame)
            else:
                logger.info("RECATCHA: reCAPTCHA v2 solved by checkbox click (no challenge).")
                # Verify if the reCAPTCHA element is gone or state changed to solved
                # This is a heuristic, actual verification might need page-specific checks
                await page.wait_for_timeout(settings.captcha_config.CAPTCHA_POST_SUBMIT_DELAY_MS) # Give time for page to update
                # A more robust check would be to see if the checkbox has changed its aria-checked state to 'true'
                # or if the main reCAPTCHA iframe has disappeared.
                try:
                    if await checkbox_element.get_attribute("aria-checked") == "true":
                        logger.info("RECATCHA: Checkbox marked as checked. Assuming solved.")
                        return True
                    # Also check for the success token if possible, though it's usually in a hidden input
                except Exception:
                    pass # Element might be detached if reCAPTCHA disappeared
                
                # If checkbox is still there and not checked, it might be a failure or a new challenge is pending
                logger.warning("RECATCHA: Checkbox click did not immediately solve reCAPTCHA or trigger challenge.")
                return False # Assume failure if not immediately solved or challenged

        except Exception as e:
            logger.error(f"RECATCHA: Error handling reCAPTCHA v2: {e}", exc_info=True)
            return False

    async def _solve_recaptcha_v2_challenge(self, challenge_frame: playwright_api.Frame, instruction: str) -> bool:
        """
        Handles the reCAPTCHA v2 image selection challenge.
        This is a complex part and will require significant development.
        """
        logger.info("RECATCHA: Attempting to solve reCAPTCHA v2 image challenge...")
        try:
            challenge_data = await self.detector.extract_recaptcha_challenge_images(challenge_frame)
            if not challenge_data:
                logger.error("RECATCHA: Failed to extract reCAPTCHA challenge images.")
                return False

            instruction = challenge_data.get("instruction", "")
            grid_images = challenge_data.get("grid_images", [])

            if not instruction or not grid_images:
                logger.warning("RECATCHA: Missing instruction or grid images for challenge.")
                return False

            logger.info(f"RECATCHA: Challenge instruction: '{instruction}'")

            # Use the vision solver to identify which tiles to click
            # This is a simplified example; actual implementation would be more complex.
            # The vision model needs to understand the instruction and apply it to each image.
            # For now, we'll just log the instruction and assume a successful click.
            
            # In a real scenario, you would loop through grid_images, send each one to
            # self.solver.solve_text_captcha (or a specialized image classification method),
            # and based on the instruction, decide if the tile should be clicked.

            clicked_tiles_count = 0
            for tile_info in grid_images:
                tile_element = tile_info["element"]
                is_target = await self.solver.classify_image_for_recaptcha(tile_info["bytes"], instruction)
                if is_target:
                    logger.info(f"RECATCHA: Clicking tile {tile_info['index']} based on vision model classification.")
                    await tile_element.click()
                    clicked_tiles_count += 1
                    await asyncio.sleep(0.5) # Small delay between clicks
                else:
                    logger.debug(f"RECATCHA: Skipping tile {tile_info['index']} as it does not match instruction.")

            logger.info(f"RECATCHA: Clicked {clicked_tiles_count} potential tiles.")

            # Click the 'Verify' button
            verify_button = await challenge_frame.query_selector("#recaptcha-verify-button")
            if verify_button and await verify_button.is_visible() and await verify_button.is_enabled():
                logger.info("RECATCHA: Clicking 'Verify' button...")
                await verify_button.click()
                await page.wait_for_timeout(settings.captcha_config.CAPTCHA_POST_SUBMIT_DELAY_MS)
                # After clicking verify, need to check if challenge is solved or reloaded
                # This part needs robust verification logic, similar to _verify_captcha_solved
                return True # Assume success for now
            else:
                logger.warning("RECATCHA: 'Verify' button not found or not interactable.")
                return False

        except Exception as e:
            logger.error(f"RECATCHA: Error solving reCAPTCHA v2 challenge: {e}", exc_info=True)
            return False

    async def check_for_captcha(self, page: playwright_api.Page) -> bool:
        try:
            captcha_info = await self.detector.find_image_captcha(page)
            return captcha_info is not None
        except Exception as e:
            logger.warning(f"HANDLER: Error checking for CAPTCHA: {e}", exc_info=True)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        success_rate = (self.total_solved / self.total_handled * 100.0) if self.total_handled > 0 else 0.0
        stats = {
            "captchas_handled": self.total_handled,
            "captchas_solved": self.total_solved,
            "handler_success_rate": round(success_rate, 2),
            "max_handler_attempts": self.max_attempts
        }
        if hasattr(self, 'solver'):
            stats.update({"solver_stats": self.solver.get_stats()})
        else:
            stats.update({"solver_stats": "Solver not initialized"})
        return stats
    
    def reset_stats(self):
        self.total_handled = 0
        self.total_solved = 0
        if hasattr(self, 'solver'):
            self.solver.solve_attempts = 0
            self.solver.solve_successes = 0
        logger.info("HANDLER: Statistics reset.")

class CaptchaIntegration:
    @staticmethod
    def add_captcha_methods(agent_class):
        def _init_captcha_handler(self_agent):
            if not hasattr(self_agent, '_captcha_handler'):
                model_override = None
                attempts_override = None
                if hasattr(self_agent, 'settings') and hasattr(self_agent.settings, 'captcha_config'):
                    cfg = self_agent.settings.captcha_config
                    model_override = cfg.CAPTCHA_VISION_MODEL_NAME
                    attempts_override = cfg.CAPTCHA_MAX_ATTEMPTS
                    logger.info(f"INTEGRATION: WebBrowserAgent using captcha_config. Model: {model_override}, Attempts: {attempts_override}")
                else:
                    logger.warning("INTEGRATION: WebBrowserAgent has no captcha_config. SimpleCaptchaHandler will use global defaults.")
                self_agent._captcha_handler = SimpleCaptchaHandler(
                    vision_model_name=model_override,
                    max_attempts_override=attempts_override
                )
        
        async def solve_captchas(self_agent) -> bool:
            self_agent._init_captcha_handler()
            if not self_agent._page:
                logger.error("INTEGRATION: No page available for CAPTCHA solving")
                return False
            return await self_agent._captcha_handler.handle_page_captchas(self_agent._page)
        
        async def check_for_captcha(self_agent) -> bool:
            self_agent._init_captcha_handler()
            if not self_agent._page:
                logger.error("INTEGRATION: No page available for CAPTCHA detection")
                return False
            return await self_agent._captcha_handler.check_for_captcha(self_agent._page)
        
        async def wait_and_solve_captcha(self_agent, page: playwright_api.Page, timeout_seconds: int = 30) -> bool:
            self_agent._init_captcha_handler()
            if not self_agent._page:
                logger.error("INTEGRATION: No page available for wait_and_solve_captcha")
                return False # Indicate failure to perform action
            
            start_time = asyncio.get_event_loop().time()
            logger.info(f"INTEGRATION: Waiting up to {timeout_seconds}s for a CAPTCHA to appear...")
            while (asyncio.get_event_loop().time() - start_time) < timeout_seconds:
                if await self_agent.check_for_captcha():
                    logger.info("INTEGRATION: CAPTCHA detected, attempting to solve...")
                    return await self_agent.solve_captchas()
                await asyncio.sleep(1)
            logger.info(f"INTEGRATION: No CAPTCHA detected within {timeout_seconds}s timeout.")
            return True # No CAPTCHA found is considered a success for this waiting function
        
        def get_captcha_stats(self_agent) -> Dict[str, Any]:
            if hasattr(self_agent, '_captcha_handler'):
                return self_agent._captcha_handler.get_stats()
            else:
                default_solver_stats = {"attempts": 0, "successes": 0, "success_rate": 0.0, "backend": "none"}
                _captcha_config = settings.captcha_config
                return {
                    "captchas_handled": 0, 
                    "captchas_solved": 0, 
                    "handler_success_rate": 0.0,
                    "max_handler_attempts": _captcha_config.CAPTCHA_MAX_ATTEMPTS,
                    "solver_stats": default_solver_stats
                }
        
        agent_class._init_captcha_handler = _init_captcha_handler
        agent_class.solve_captchas = solve_captchas
        agent_class.check_for_captcha = check_for_captcha
        agent_class.wait_and_solve_captcha = wait_and_solve_captcha
        agent_class.get_captcha_stats = get_captcha_stats
        
        logger.info("INTEGRATION: CAPTCHA methods successfully added to WebBrowserAgent.")
        return agent_class

# Example usage (commented out)
# if __name__ == "__main__":
#     async def test_handler():
#         from playwright.async_api import async_playwright
#         # Ensure .env is loaded if settings rely on it and run from here
#         # from dotenv import load_dotenv
#         # load_dotenv()
#         handler = SimpleCaptchaHandler() # Will use global settings
#         async with async_playwright() as p:
#             browser = await p.chromium.launch(headless=False)
#             page = await browser.new_page()
#             # Replace with an actual CAPTCHA test page URL
#             await page.goto("https://www.example.com/captcha_test_page") 
#             success = await handler.handle_page_captchas(page)
#             print(f"CAPTCHA solving result: {success}")
#             print(f"Stats: {handler.get_stats()}")
#             await browser.close()
#     # asyncio.run(test_handler())

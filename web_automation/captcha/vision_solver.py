# web_automation/captcha/vision_solver.py
import asyncio
import logging
import base64
import re
from typing import Optional, Dict
from .exceptions import VisionModelUnavailable, CaptchaSolveFailed

logger = logging.getLogger(__name__)

class OllamaVisionSolver:
    """
    Local vision model integration for solving image CAPTCHAs.
    Supports multiple backends: Ollama, Transformers, OCR fallback.
    """
    
    def __init__(self, model_name: str = "llava:7b"):
        self.model_name = model_name
        self.backend = self._detect_available_backend()
        self.solve_attempts = 0
        self.solve_successes = 0
        
    def _detect_available_backend(self) -> str:
        """Auto-detect which vision backend is available."""
        # Try Ollama first (most likely for local models)
        try:
            import ollama
            # Test if ollama is running
            ollama.list()
            logger.info("VISION: Ollama backend available")
            return "ollama"
        except:
            pass
            
        # Try Transformers (HuggingFace)
        try:
            import transformers
            logger.info("VISION: Transformers backend available")
            return "transformers"
        except ImportError:
            pass
            
        # Fallback to OCR
        try:
            import pytesseract
            logger.info("VISION: OCR fallback backend available")
            return "ocr"
        except ImportError:
            pass
            
        logger.warning("VISION: No vision backends available")
        return "none"
    
    def is_model_available(self) -> bool:
        """Check if vision model is loaded and ready."""
        return self.backend != "none"
    
    async def solve_text_captcha(self, image_bytes: bytes) -> Optional[str]:
        """Solve text-based CAPTCHA using vision model."""
        if not self.is_model_available():
            raise VisionModelUnavailable("No vision backend available")
            
        self.solve_attempts += 1
        
        try:
            # Preprocess image for better recognition
            processed_image = await self._preprocess_image(image_bytes)
            
            # Try vision model first
            if self.backend == "ollama":
                result = await self._solve_with_ollama(processed_image)
            elif self.backend == "transformers":
                result = await self._solve_with_transformers(processed_image)
            elif self.backend == "ocr":
                result = await self._solve_with_ocr(processed_image)
            else:
                raise VisionModelUnavailable("No valid backend")
                
            if result:
                self.solve_successes += 1
                logger.info(f"VISION: Successfully solved CAPTCHA: '{result}'")
                return result
            else:
                logger.warning("VISION: Model returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"VISION: Error solving CAPTCHA: {e}")
            raise CaptchaSolveFailed(f"Vision solving failed: {e}")
    
    async def solve_math_captcha(self, image_bytes: bytes) -> Optional[str]:
        """Solve simple math CAPTCHAs."""
        result = await self.solve_text_captcha(image_bytes)
        if result:
            # Try to extract and evaluate math expressions
            math_result = self._evaluate_math_expression(result)
            return math_result or result
        return None
    
    async def analyze_captcha_type(self, image_bytes: bytes) -> str:
        """Determine what type of CAPTCHA this is."""
        # Simple heuristic - could be improved with actual analysis
        result = await self.solve_text_captcha(image_bytes)
        if result:
            if any(op in result for op in ['+', '-', '*', '=', 'x']):
                return "math"
            elif result.replace(' ', '').isalnum():
                return "text"
        return "unknown"
    
    async def _solve_with_ollama(self, image_bytes: bytes) -> Optional[str]:
        """Solve using Ollama vision model."""
        try:
            import ollama
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            prompt = "What text, letters, or numbers do you see in this CAPTCHA image? Return only the text you see, nothing else."
            
            response = await asyncio.to_thread(
                ollama.generate,
                model=self.model_name,
                prompt=prompt,
                images=[image_b64]
            )
            
            if response and 'response' in response:
                result = response['response'].strip()
                return self._clean_captcha_result(result)
                
        except Exception as e:
            logger.error(f"VISION: Ollama solving failed: {e}")
            
        return None
    
    async def _solve_with_transformers(self, image_bytes: bytes) -> Optional[str]:
        """Solve using HuggingFace Transformers."""
        try:
            from transformers import pipeline
            from PIL import Image
            import io
            
            # Load vision model (this should be cached after first load)
            if not hasattr(self, '_vision_pipeline'):
                self._vision_pipeline = pipeline(
                    "image-to-text", 
                    model="nlpconnect/vit-gpt2-image-captioning"
                )
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate caption and extract text
            result = await asyncio.to_thread(self._vision_pipeline, image)
            if result and len(result) > 0:
                text = result[0].get('generated_text', '')
                return self._clean_captcha_result(text)
                
        except Exception as e:
            logger.error(f"VISION: Transformers solving failed: {e}")
            
        return None
    
    async def _solve_with_ocr(self, image_bytes: bytes) -> Optional[str]:
        """Fallback OCR solution using Tesseract."""
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Configure Tesseract for CAPTCHA-like text
            custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
            
            result = await asyncio.to_thread(
                pytesseract.image_to_string, 
                image, 
                config=custom_config
            )
            
            if result:
                return self._clean_captcha_result(result)
                
        except Exception as e:
            logger.error(f"VISION: OCR solving failed: {e}")
            
        return None

    async def classify_image_for_recaptcha(self, image_bytes: bytes, instruction: str) -> bool:
        """
        Classifies an image tile for reCAPTCHA based on the given instruction.
        Returns True if the image contains the object specified in the instruction, False otherwise.
        """
        if not self.is_model_available():
            raise VisionModelUnavailable("No vision backend available for reCAPTCHA classification")

        try:
            processed_image = await self._preprocess_image(image_bytes)
            if self.backend == "ollama":
                # Construct a prompt that asks the model to identify if the image contains the object
                # specified in the instruction.
                prompt = f"Does this image contain any of the following: {instruction}? Answer only with 'yes' or 'no'."
                
                image_b64 = base64.b64encode(processed_image).decode('utf-8')
                response = await asyncio.to_thread(
                    ollama.generate,
                    model=self.model_name,
                    prompt=prompt,
                    images=[image_b64]
                )
                
                if response and 'response' in response:
                    result = response['response'].strip().lower()
                    logger.debug(f"VISION: reCAPTCHA classification result for '{instruction}': {result}")
                    return result == 'yes'
                
            # Add logic for other backends if necessary, or raise an error
            logger.warning(f"VISION: reCAPTCHA classification not implemented for backend: {self.backend}")
            return False

        except Exception as e:
            logger.error(f"VISION: Error classifying reCAPTCHA image: {e}", exc_info=True)
            return False
    
    async def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """Clean up CAPTCHA image for better recognition."""
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            import io
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
            
            # Resize if too small (helps with recognition)
            width, height = image.size
            if width < 100 or height < 50:
                scale = max(100/width, 50/height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save back to bytes
            output = io.BytesIO()
            image.save(output, format='PNG')
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"VISION: Image preprocessing failed, using original: {e}")
            return image_bytes
    
    def _clean_captcha_result(self, raw_result: str) -> Optional[str]:
        """Clean and validate CAPTCHA solution."""
        if not raw_result:
            return None
            
        # Remove common noise
        cleaned = raw_result.strip()
        cleaned = re.sub(r'[^\w\s+\-*/=]', '', cleaned)  # Keep alphanumeric and math ops
        cleaned = cleaned.replace('\n', '').replace('\r', '')
        
        # Remove common OCR artifacts
        cleaned = cleaned.replace('|', 'I').replace('0', 'O')  # Common confusions
        
        # Validate result length (most CAPTCHAs are 3-8 characters)
        if len(cleaned) < 2 or len(cleaned) > 10:
            logger.warning(f"VISION: Result length suspicious: '{cleaned}' ({len(cleaned)} chars)")
            return None
            
        return cleaned if cleaned else None
    
    def _evaluate_math_expression(self, expression: str) -> Optional[str]:
        """Safely evaluate simple math expressions."""
        try:
            # Clean the expression
            expr = expression.replace('x', '*').replace('X', '*')
            expr = re.sub(r'[^\d+\-*/\s=]', '', expr)
            
            # Extract the part before '=' if present
            if '=' in expr:
                expr = expr.split('=')[0]
            
            # Simple validation - only allow basic math
            if re.match(r'^[\d+\-*/\s]+$', expr):
                result = eval(expr)  # Safe for simple math expressions
                return str(int(result))
                
        except Exception as e:
            logger.debug(f"VISION: Math evaluation failed for '{expression}': {e}")
            
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Return solving statistics."""
        success_rate = (self.solve_successes / self.solve_attempts * 100) if self.solve_attempts > 0 else 0
        return {
            "attempts": self.solve_attempts,
            "successes": self.solve_successes,
            "success_rate": round(success_rate, 1),
            "backend": self.backend
        }

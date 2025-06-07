"""
Standalone image analysis utility for explicit visual processing.
Provides powerful image analysis capabilities without automatic storage overhead.
"""

import base64
import os
from typing import Optional, Union
import ollama
from ollama import AsyncClient

class ImageAnalyzer:
    """
    Standalone image analysis utility for on-demand visual processing.
    
    Provides LLM-powered image analysis without automatic memory storage.
    Supports image description, OCR, UI element identification, and comparison.
    """
    
    def __init__(self, model_name: str = os.getenv("VISUAL_SYSTEM_MODEL", "qwen2.5vl:7b"), ollama_base_url: str = "http://localhost:11434"):
        """Initialize ImageAnalyzer with specified LLM model."""
        self.model_name = model_name
        self.llm_client = AsyncClient(host=ollama_base_url)
    
    async def analyze_image(self, image_data: Union[bytes, str], prompt: str = "Describe what you see in this image") -> str:
        """
        Analyze single image with custom prompt.
        
        Args:
            image_data: Image as bytes, file path, or base64 string
            prompt: Analysis prompt for the LLM
            
        Returns:
            LLM analysis response as string
        """
        image_base64 = await self._prepare_image_data(image_data)
        
        response = await self.llm_client.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_base64]
            }]
        )
        
        return response['message']['content']
    
    async def extract_text_from_image(self, image_data: Union[bytes, str]) -> str:
        """Extract all text from image using OCR capabilities."""
        return await self.analyze_image(
            image_data, 
            "Extract all text from this image. Maintain formatting and structure. Only return the extracted text."
        )
    
    async def identify_ui_elements(self, image_data: Union[bytes, str]) -> str:
        """Identify interactive UI elements in screenshot."""
        return await self.analyze_image(
            image_data,
            "Identify all clickable elements, buttons, forms, links, and interactive components in this interface. Describe their locations and purposes."
        )
    
    async def compare_images(self, image1: Union[bytes, str], image2: Union[bytes, str], 
                           comparison_prompt: Optional[str] = None) -> str:
        """
        Compare two images for similarities and differences.
        
        Args:
            image1: First image for comparison
            image2: Second image for comparison  
            comparison_prompt: Custom comparison prompt
            
        Returns:
            LLM comparison analysis
        """
        prompt = comparison_prompt or "Compare these two images. Describe the key differences and similarities."
        
        image1_base64 = await self._prepare_image_data(image1)
        image2_base64 = await self._prepare_image_data(image2)
        
        response = await self.llm_client.chat(
            model=self.model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image1_base64, image2_base64]
            }]
        )
        
        return response['message']['content']
    
    async def analyze_page_layout(self, image_data: Union[bytes, str]) -> str:
        """Analyze webpage layout and structure."""
        return await self.analyze_image(
            image_data,
            "Analyze this webpage layout. Describe the structure, navigation elements, main content areas, and overall design patterns."
        )
    
    async def detect_errors_or_changes(self, image_data: Union[bytes, str]) -> str:
        """Detect error messages, alerts, or significant UI changes."""
        return await self.analyze_image(
            image_data,
            "Look for error messages, alerts, notifications, loading states, or any unusual UI elements that might indicate problems or state changes."
        )
    
    async def _prepare_image_data(self, image_data: Union[bytes, str]) -> str:
        """Convert various image formats to base64 string."""
        if isinstance(image_data, bytes):
            return base64.b64encode(image_data).decode('utf-8')
        elif isinstance(image_data, str):
            # Check if it's a file path
            if os.path.exists(image_data):
                with open(image_data, 'rb') as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            else:
                # Assume it's already base64
                return image_data
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

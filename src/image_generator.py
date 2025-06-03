import os
import io
import requests
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient, login
import cv2

class ImageGenerator:
    def __init__(self):
        """Initialize the image generator with API settings."""
        load_dotenv()  # Load environment variables
        self.api_token = os.getenv('HUGGINGFACE_TOKEN')
        if not self.api_token:
            raise ValueError("Hugging Face token not found. Please set HUGGINGFACE_TOKEN in .env file")
        
        try:
            # Verify token by attempting to log in
            login(token=self.api_token)
            
            # Initialize the client with a specific model
            self.client = InferenceClient(
                "CompVis/stable-diffusion-v1-4",  # Using a different model that's more reliable
                token=self.api_token
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Hugging Face client. Error: {str(e)}")

    def generate_from_prompt(self, prompt: str, negative_prompt: str = None) -> np.ndarray:
        """
        Generate an image from a text prompt using Stable Diffusion.
        
        Args:
            prompt: Text description of the desired image
            negative_prompt: Things to avoid in the generated image
            
        Returns:
            Generated image as numpy array in BGR format (for OpenCV)
        """
        try:
            # Add some context to the prompt for better face generation
            enhanced_prompt = f"high quality, detailed portrait, {prompt}"
            if not negative_prompt:
                negative_prompt = "blurry, low quality, distorted, deformed"
                
            # Generate the image
            image_data = self.client.text_to_image(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                model="CompVis/stable-diffusion-v1-4",
                width=512,
                height=512
            )
            
            # Convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            # Convert to BGR for OpenCV
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            raise Exception(f"Failed to generate image: {str(e)}\nPlease verify your Hugging Face token is valid and has the necessary permissions.")

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance the user's prompt with additional context for better results.
        
        Args:
            prompt: Original user prompt
            
        Returns:
            Enhanced prompt with additional context
        """
        # Add style-related keywords
        style_keywords = [
            "professional photography",
            "studio lighting",
            "high resolution",
            "detailed facial features",
            "sharp focus"
        ]
        
        # Combine original prompt with style keywords
        enhanced = f"{prompt}, {', '.join(style_keywords)}"
        return enhanced 
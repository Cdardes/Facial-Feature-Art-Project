import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from environment variables or user input"""
    load_dotenv()  # Load environment variables from .env file if it exists
    
    # Get token from environment variable or ask user
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        print("Please enter your Hugging Face token:")
        token = input().strip()
        # Store for current session
        os.environ['HUGGINGFACE_TOKEN'] = token
    
    return {
        'huggingface_token': token
    }

def get_huggingface_token():
    """Get the Hugging Face token"""
    config = load_config()
    return config['huggingface_token'] 
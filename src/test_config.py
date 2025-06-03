from .config import get_huggingface_token

def test_token():
    """Test that we can retrieve the Hugging Face token"""
    token = get_huggingface_token()
    if token:
        print("✓ Token successfully loaded!")
        print(f"Token starts with: {token[:8]}...")
    else:
        print("✗ Failed to load token!")

if __name__ == "__main__":
    test_token() 
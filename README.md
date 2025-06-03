# Facial Feature Art Project

This project uses computer vision techniques to detect facial features and create artistic representations of faces using OpenCV and Python.

## Features
- Facial landmark detection
- Artistic filters and transformations
- Interactive image manipulation
- Support for various artistic styles

## Setup
1. Install Python 3.8 or higher
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the shape predictor file:
   - Download the shape predictor from [dlib's official model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Extract and place it in the `models` directory

## Configuration

### Hugging Face Token
This project uses Hugging Face models for enhanced facial feature detection. You need to provide your Hugging Face token in one of two ways:

1. Create a `.env` file in the project root with:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```

2. Or enter the token when prompted during application startup.

To get a Hugging Face token:
1. Sign up at https://huggingface.co
2. Go to your profile settings
3. Navigate to "Access Tokens"
4. Create a new token with read access

## Usage
1. Run the main application:
   ```
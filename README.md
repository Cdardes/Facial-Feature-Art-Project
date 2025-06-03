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

## Usage
1. Run the main application:
   ```
   python src/main.py
   ```
2. Select an image using the file dialog
3. Choose artistic filters and effects
4. Save your creation

## Project Structure
```
├── src/
│   ├── main.py
│   ├── face_detector.py
│   └── art_filters.py
├── models/
│   └── shape_predictor_68_face_landmarks.dat
├── samples/
│   └── example.jpg
└── output/
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

import cv2
import numpy as np
from typing import Tuple, List

class FaceDetector:
    def __init__(self):
        """Initialize the face detector with OpenCV's pre-trained model."""
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect_features(self, image: np.ndarray) -> dict:
        """
        Detect facial features in the given image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Dictionary containing detected facial features (faces, eyes, mouth)
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        features = []
        for (x, y, w, h) in faces:
            face_dict = {
                'face': (x, y, w, h),
                'eyes': [],
                'mouth': []
            }
            
            # Get the face ROI (Region of Interest)
            roi_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                face_dict['eyes'].append((x+ex, y+ey, ew, eh))
            
            # Detect mouth
            mouth = self.mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            for (mx, my, mw, mh) in mouth:
                face_dict['mouth'].append((x+mx, y+my, mw, mh))
                
            features.append(face_dict)
            
        return features

    def draw_features(self, image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Draw detected facial features on the image.
        
        Args:
            image: Input image
            features: List of feature dictionaries
            
        Returns:
            Image with drawn features
        """
        img_copy = image.copy()
        
        for face_features in features:
            # Draw face rectangle
            x, y, w, h = face_features['face']
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw eyes
            for (ex, ey, ew, eh) in face_features['eyes']:
                cv2.rectangle(img_copy, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # Draw mouth
            for (mx, my, mw, mh) in face_features['mouth']:
                cv2.rectangle(img_copy, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                
        return img_copy 
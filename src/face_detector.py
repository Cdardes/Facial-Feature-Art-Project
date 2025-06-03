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

def detect_facial_landmarks(image):
    """
    Detect facial landmarks using OpenCV's face detection
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        return None
    
    # Get the first face detected
    (x, y, w, h) = faces[0]
    
    # Create basic facial landmarks (simplified version)
    landmarks = {
        'jaw': [(x + int(w*0.2), y + h), (x + int(w*0.8), y + h)],
        'right_eyebrow': [(x + int(w*0.2), y + int(h*0.3)), (x + int(w*0.4), y + int(h*0.3))],
        'left_eyebrow': [(x + int(w*0.6), y + int(h*0.3)), (x + int(w*0.8), y + int(h*0.3))],
        'nose_bridge': [(x + int(w*0.5), y + int(h*0.4)), (x + int(w*0.5), y + int(h*0.6))],
        'nose_tip': [(x + int(w*0.4), y + int(h*0.6)), (x + int(w*0.6), y + int(h*0.6))],
        'right_eye': [(x + int(w*0.25), y + int(h*0.4)), (x + int(w*0.35), y + int(h*0.4))],
        'left_eye': [(x + int(w*0.65), y + int(h*0.4)), (x + int(w*0.75), y + int(h*0.4))],
        'outer_lip': [(x + int(w*0.3), y + int(h*0.8)), (x + int(w*0.7), y + int(h*0.8))]
    }
    
    return landmarks

def draw_landmarks(image, landmarks):
    """
    Draw facial landmarks on the image
    """
    if landmarks is None:
        return image
    
    img_copy = image.copy()
    
    # Draw each facial feature
    for feature, points in landmarks.items():
        for point in points:
            cv2.circle(img_copy, point, 2, (0, 255, 0), -1)
        # Connect points if there are multiple
        if len(points) > 1:
            cv2.line(img_copy, points[0], points[1], (0, 255, 0), 1)
    
    return img_copy 
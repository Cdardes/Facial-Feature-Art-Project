import cv2
import dlib
import numpy as np
from typing import Tuple, List

class FaceDetector:
    def __init__(self, predictor_path: str):
        """Initialize the face detector with the shape predictor model."""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_landmarks(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect facial landmarks in the given image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected facial landmarks as numpy arrays
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        landmarks_list = []
        
        for face in faces:
            # Predict facial landmarks
            landmarks = self.predictor(gray, face)
            # Convert landmarks to numpy array
            landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])
            landmarks_list.append(landmarks_points)
            
        return landmarks_list

    def get_facial_features(self, landmarks: np.ndarray) -> dict:
        """
        Extract specific facial features from landmarks.
        
        Args:
            landmarks: Array of facial landmarks
            
        Returns:
            Dictionary containing different facial features
        """
        features = {
            'jaw': landmarks[0:17],
            'right_eyebrow': landmarks[17:22],
            'left_eyebrow': landmarks[22:27],
            'nose_bridge': landmarks[27:31],
            'nose_tip': landmarks[31:36],
            'right_eye': landmarks[36:42],
            'left_eye': landmarks[42:48],
            'outer_lips': landmarks[48:60],
            'inner_lips': landmarks[60:68]
        }
        return features

    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray, color=(0, 255, 0), size=3):
        """
        Draw facial landmarks on the image.
        
        Args:
            image: Input image
            landmarks: Array of facial landmarks
            color: BGR color tuple for drawing
            size: Size of landmark points
        """
        for point in landmarks:
            cv2.circle(image, tuple(point), size, color, -1) 
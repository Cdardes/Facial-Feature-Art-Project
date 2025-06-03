import cv2
import numpy as np
from typing import Tuple, List

class ArtisticFilters:
    @staticmethod
    def apply_sketch_effect(image: np.ndarray) -> np.ndarray:
        """
        Convert image to a pencil sketch effect.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Laplacian(blur, cv2.CV_8U, ksize=5)
        ret, sketch = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_cartoon_effect(image: np.ndarray) -> np.ndarray:
        """
        Apply cartoon effect to the image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    @staticmethod
    def apply_pop_art(image: np.ndarray) -> List[np.ndarray]:
        """
        Create pop art style effect with multiple color variations.
        """
        pop_arts = []
        
        # Create different color variations
        color_variations = [
            {'b': 0.8, 'g': 0.2, 'r': 0.2},
            {'b': 0.2, 'g': 0.8, 'r': 0.2},
            {'b': 0.2, 'g': 0.2, 'r': 0.8},
            {'b': 0.8, 'g': 0.8, 'r': 0.2}
        ]
        
        for color in color_variations:
            temp = image.copy()
            temp[:, :, 0] = temp[:, :, 0] * color['b']
            temp[:, :, 1] = temp[:, :, 1] * color['g']
            temp[:, :, 2] = temp[:, :, 2] * color['r']
            
            # Enhance contrast
            temp = cv2.convertScaleAbs(temp, alpha=1.3, beta=20)
            pop_arts.append(temp)
            
        return pop_arts

    @staticmethod
    def apply_feature_highlight(image: np.ndarray, landmarks: np.ndarray, 
                              feature_name: str, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Highlight specific facial features with artistic effects.
        """
        result = image.copy()
        mask = np.zeros_like(image)
        
        # Draw the feature on the mask
        cv2.fillPoly(mask, [landmarks], color)
        
        # Blend the mask with the original image
        result = cv2.addWeighted(result, 0.8, mask, 0.2, 0)
        return result

    @staticmethod
    def create_abstract_art(landmarks: np.ndarray, image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Create abstract art based on facial landmarks.
        """
        canvas = np.zeros(image_shape, dtype=np.uint8)
        
        # Generate random colors
        colors = np.random.randint(0, 255, (len(landmarks), 3))
        
        # Draw abstract shapes based on landmarks
        for i, point in enumerate(landmarks):
            size = np.random.randint(5, 20)
            color = tuple(map(int, colors[i]))
            cv2.circle(canvas, tuple(point), size, color, -1)
            
            if i > 0:
                cv2.line(canvas, tuple(landmarks[i-1]), tuple(point), color, 2)
        
        return canvas 
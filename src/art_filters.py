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
    def apply_pop_art_effect(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create a pop art effect focusing on facial features.
        """
        output = image.copy()
        
        # Apply vibrant color palette
        output = cv2.convertScaleAbs(output, alpha=1.2, beta=10)
        
        for face_dict in features:
            # Get face region
            x, y, w, h = face_dict['face']
            face_roi = output[y:y+h, x:x+w]
            
            # Apply color effects to eyes
            for (ex, ey, ew, eh) in face_dict['eyes']:
                # Convert to local coordinates
                local_ex, local_ey = ex-x, ey-y
                cv2.rectangle(face_roi, (local_ex, local_ey), 
                            (local_ex+ew, local_ey+eh), (0, 255, 255), -1)
            
            # Apply color effects to mouth
            for (mx, my, mw, mh) in face_dict['mouth']:
                # Convert to local coordinates
                local_mx, local_my = mx-x, my-y
                cv2.rectangle(face_roi, (local_mx, local_my), 
                            (local_mx+mw, local_my+mh), (255, 50, 50), -1)
            
            # Add artistic border to face
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
        return output

    @staticmethod
    def apply_abstract_art(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create an abstract art representation of facial features.
        """
        # Create a black canvas
        canvas = np.zeros_like(image)
        
        for face_dict in features:
            # Draw face outline
            x, y, w, h = face_dict['face']
            cv2.ellipse(canvas, (x + w//2, y + h//2), (w//2, h//2), 
                       0, 0, 360, (255, 255, 255), -1)
            
            # Draw abstract eyes
            for (ex, ey, ew, eh) in face_dict['eyes']:
                center = (ex + ew//2, ey + eh//2)
                cv2.circle(canvas, center, max(ew, eh)//2, (0, 255, 255), -1)
                cv2.circle(canvas, center, max(ew, eh)//4, (255, 0, 0), -1)
            
            # Draw abstract mouth
            for (mx, my, mw, mh) in face_dict['mouth']:
                cv2.ellipse(canvas, (mx + mw//2, my + mh//2), 
                          (mw//2, mh//2), 0, 0, 180, (0, 0, 255), -1)
        
        # Add some artistic effects
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        canvas = cv2.addWeighted(canvas, 0.7, image, 0.3, 0)
        
        return canvas

    @staticmethod
    def apply_feature_highlighting(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create an artistic effect that highlights facial features.
        """
        # Create a darkened version of the image
        output = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
        
        for face_dict in features:
            # Get face region
            x, y, w, h = face_dict['face']
            face_roi = image[y:y+h, x:x+w]
            
            # Create mask for the face
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
            
            # Copy the original face with mask
            output[y:y+h, x:x+w] = cv2.bitwise_and(face_roi, face_roi, mask=mask)
            
            # Highlight eyes
            for (ex, ey, ew, eh) in face_dict['eyes']:
                # Convert to local coordinates
                local_ex, local_ey = ex-x, ey-y
                eye_roi = face_roi[local_ey:local_ey+eh, local_ex:local_ex+ew]
                # Brighten the eyes
                bright_eyes = cv2.convertScaleAbs(eye_roi, alpha=1.5, beta=30)
                face_roi[local_ey:local_ey+eh, local_ex:local_ex+ew] = bright_eyes
            
            # Highlight mouth
            for (mx, my, mw, mh) in face_dict['mouth']:
                # Convert to local coordinates
                local_mx, local_my = mx-x, my-y
                mouth_roi = face_roi[local_my:local_my+mh, local_mx:local_mx+mw]
                # Add color to the lips
                colored_lips = cv2.addWeighted(mouth_roi, 0.7, 
                                             np.full_like(mouth_roi, (80, 0, 255)), 0.3, 0)
                face_roi[local_my:local_my+mh, local_mx:local_mx+mw] = colored_lips
        
        return output 
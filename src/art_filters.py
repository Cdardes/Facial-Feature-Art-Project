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

    @staticmethod
    def apply_neon_glow(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create a neon glow effect around facial features.
        """
        # Create a dark background
        output = np.zeros_like(image)
        
        for face_dict in features:
            # Get face region
            x, y, w, h = face_dict['face']
            
            # Create neon face outline
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.ellipse(mask, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, 255, 2)
            
            # Apply neon glow effect
            blur_amount = 15
            mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
            output[:,:,0] = cv2.add(output[:,:,0], mask * 0.5)  # Blue
            output[:,:,1] = cv2.add(output[:,:,1], mask)        # Green
            
            # Add neon eyes
            for (ex, ey, ew, eh) in face_dict['eyes']:
                center = (ex + ew//2, ey + eh//2)
                cv2.circle(output, center, ew//2, (0, 255, 255), 2)  # Yellow
                cv2.circle(output, center, ew//4, (255, 255, 255), -1)  # White
            
            # Add neon mouth
            for (mx, my, mw, mh) in face_dict['mouth']:
                cv2.ellipse(output, (mx + mw//2, my + mh//2), 
                          (mw//2, mh//3), 0, 0, 180, (0, 0, 255), 2)  # Red
        
        # Add some of the original image for context
        output = cv2.addWeighted(output, 0.7, image, 0.3, 0)
        return output

    @staticmethod
    def apply_mosaic_art(image: np.ndarray, features: List[dict], cell_size: int = 15) -> np.ndarray:
        """
        Create a mosaic art effect with emphasis on facial features.
        """
        output = image.copy()
        height, width = image.shape[:2]
        
        # Create mosaic effect on the entire image
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                roi = output[y:min(y+cell_size, height), x:min(x+cell_size, width)]
                if roi.size > 0:
                    color = roi.mean(axis=(0, 1))
                    roi[:] = color
        
        # Enhance facial features
        for face_dict in features:
            x, y, w, h = face_dict['face']
            
            # Smaller cell size for face region
            face_cell_size = cell_size // 2
            face_roi = output[y:y+h, x:x+w]
            
            for fy in range(0, h, face_cell_size):
                for fx in range(0, w, face_cell_size):
                    roi = face_roi[fy:min(fy+face_cell_size, h), fx:min(fx+face_cell_size, w)]
                    if roi.size > 0:
                        color = roi.mean(axis=(0, 1))
                        roi[:] = color
            
            # Add colored borders around features
            cv2.rectangle(output, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            for (ex, ey, ew, eh) in face_dict['eyes']:
                cv2.rectangle(output, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            
            for (mx, my, mw, mh) in face_dict['mouth']:
                cv2.rectangle(output, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
        
        return output

    @staticmethod
    def apply_watercolor(image: np.ndarray) -> np.ndarray:
        """
        Create a watercolor painting effect.
        """
        # Apply bilateral filter for smoothing while preserving edges
        smooth = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Reduce color palette
        num_colors = 8
        colors = np.float32(smooth).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(colors, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        reduced = res.reshape(image.shape)
        
        # Add texture
        texture = np.zeros_like(image)
        for i in range(3):
            texture[:,:,i] = cv2.GaussianBlur(np.random.normal(0, 50, image.shape[:2]), (21, 21), 0)
        
        # Blend with texture
        result = cv2.addWeighted(reduced, 0.9, texture, 0.1, 0)
        return result

    @staticmethod
    def apply_comic_book(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create a comic book style effect.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create edge mask
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, None)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Reduce color palette
        num_colors = 6
        colors = np.float32(image).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(colors, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        reduced = res.reshape(image.shape)
        
        # Combine edges with reduced colors
        result = cv2.subtract(reduced, edges)
        
        # Add comic-style dots
        for face_dict in features:
            x, y, w, h = face_dict['face']
            # Add halftone-like dots in shadow areas
            face_gray = gray[y:y+h, x:x+w]
            dots = np.zeros_like(face_gray)
            dots[face_gray < 128] = 255
            dots = cv2.erode(dots, np.ones((3,3), np.uint8))
            
            # Add dots to the result
            result[y:y+h, x:x+w][dots == 255] = (0, 0, 0)
        
        return result

    @staticmethod
    def apply_double_exposure(image: np.ndarray, features: List[dict]) -> np.ndarray:
        """
        Create a double exposure effect using facial features.
        """
        if not features:
            return image
            
        # Create a gradient background
        height, width = image.shape[:2]
        gradient = np.zeros((height, width), dtype=np.uint8)
        for i in range(width):
            gradient[:, i] = int(255 * i / width)
        gradient = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        
        # Create a mask from facial features
        mask = np.zeros((height, width), dtype=np.uint8)
        for face_dict in features:
            x, y, w, h = face_dict['face']
            cv2.ellipse(mask, (x + w//2, y + h//2), (w//2, h//2), 
                       0, 0, 360, 255, -1)
            
            # Add eyes to mask
            for (ex, ey, ew, eh) in face_dict['eyes']:
                cv2.circle(mask, (ex + ew//2, ey + eh//2), ew//2, 255, -1)
        
        # Blur the mask
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        # Create the double exposure effect
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = image.copy()
        result = image * mask_3channel + gradient * (1 - mask_3channel)
        result = result.astype(np.uint8)
        
        return result 
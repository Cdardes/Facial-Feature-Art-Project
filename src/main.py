import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
from face_detector import FaceDetector
from art_filters import ArtisticFilters

class FacialFeatureArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Feature Art Creator")
        
        # Initialize detector
        model_path = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
        if not os.path.exists(model_path):
            tk.messagebox.showerror("Error", "Please download the shape predictor model file!")
            return
            
        self.detector = FaceDetector(model_path)
        self.art_filters = ArtisticFilters()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        
        # Create filter selection
        ttk.Label(self.main_frame, text="Select Effect:").grid(row=1, column=0, padx=5, pady=5)
        self.effect_var = tk.StringVar()
        effects = ['Sketch', 'Cartoon', 'Pop Art', 'Feature Highlight', 'Abstract Art']
        self.effect_combo = ttk.Combobox(self.main_frame, textvariable=self.effect_var, values=effects)
        self.effect_combo.grid(row=1, column=1, padx=5, pady=5)
        self.effect_combo.set(effects[0])
        
        # Apply button
        ttk.Button(self.main_frame, text="Apply Effect", command=self.apply_effect).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Save button
        ttk.Button(self.main_frame, text="Save Image", command=self.save_image).grid(row=2, column=2, pady=10)
        
        # Image display
        self.canvas = tk.Canvas(self.main_frame, width=600, height=400)
        self.canvas.grid(row=3, column=0, columnspan=3, pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image)
            
    def display_image(self, image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to fit canvas while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        canvas_width = 600
        canvas_height = 400
        
        # Calculate scaling factor
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image_resized))
        
        # Update canvas
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width//2, new_height//2, image=self.photo)
        
    def apply_effect(self):
        if not hasattr(self, 'original_image'):
            tk.messagebox.showerror("Error", "Please load an image first!")
            return
            
        # Detect landmarks
        landmarks_list = self.detector.detect_landmarks(self.original_image)
        if not landmarks_list:
            tk.messagebox.showerror("Error", "No face detected in the image!")
            return
            
        landmarks = landmarks_list[0]  # Use first face detected
        
        # Apply selected effect
        effect = self.effect_var.get()
        if effect == 'Sketch':
            result = self.art_filters.apply_sketch_effect(self.original_image)
        elif effect == 'Cartoon':
            result = self.art_filters.apply_cartoon_effect(self.original_image)
        elif effect == 'Pop Art':
            results = self.art_filters.apply_pop_art(self.original_image)
            result = np.vstack([np.hstack(results[:2]), np.hstack(results[2:])])
        elif effect == 'Feature Highlight':
            features = self.detector.get_facial_features(landmarks)
            result = self.original_image.copy()
            colors = {
                'left_eye': (0, 255, 0),
                'right_eye': (0, 255, 0),
                'nose_tip': (255, 0, 0),
                'outer_lips': (0, 0, 255)
            }
            for feature_name, color in colors.items():
                result = self.art_filters.apply_feature_highlight(
                    result, features[feature_name], feature_name, color)
        else:  # Abstract Art
            result = self.art_filters.create_abstract_art(
                landmarks, self.original_image.shape)
            
        self.current_result = result
        self.display_image(result)
        
    def save_image(self):
        if not hasattr(self, 'current_result'):
            tk.messagebox.showerror("Error", "Please apply an effect first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.current_result)
            tk.messagebox.showinfo("Success", "Image saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialFeatureArtApp(root)
    root.mainloop() 
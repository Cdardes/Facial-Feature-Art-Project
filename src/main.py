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
        
        # Initialize detector and filters
        self.detector = FaceDetector()
        self.art_filters = ArtisticFilters()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create buttons
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        
        # Create filter selection
        ttk.Label(self.main_frame, text="Select Art Style:").grid(row=1, column=0, padx=5, pady=5)
        self.style_var = tk.StringVar(value="sketch")
        styles = ["sketch", "cartoon", "pop_art", "abstract", "highlight"]
        self.style_combo = ttk.Combobox(self.main_frame, textvariable=self.style_var, values=styles)
        self.style_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Create apply button
        ttk.Button(self.main_frame, text="Apply Effect", command=self.apply_effect).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create save button
        ttk.Button(self.main_frame, text="Save Image", command=self.save_image).grid(row=2, column=2, pady=10)
        
        # Create image display areas
        self.original_label = ttk.Label(self.main_frame, text="Original Image")
        self.original_label.grid(row=3, column=0, padx=5, pady=5)
        
        self.processed_label = ttk.Label(self.main_frame, text="Processed Image")
        self.processed_label.grid(row=3, column=1, padx=5, pady=5)
        
        # Initialize variables
        self.current_image = None
        self.processed_image = None
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            # Load and display the image
            self.current_image = cv2.imread(file_path)
            if self.current_image is not None:
                self.display_image(self.current_image, self.original_label)
                
    def apply_effect(self):
        if self.current_image is None:
            return
            
        # Detect facial features
        features = self.detector.detect_features(self.current_image)
        
        # Apply selected effect
        style = self.style_var.get()
        if style == "sketch":
            self.processed_image = self.art_filters.apply_sketch_effect(self.current_image)
        elif style == "cartoon":
            self.processed_image = self.art_filters.apply_cartoon_effect(self.current_image)
        elif style == "pop_art":
            self.processed_image = self.art_filters.apply_pop_art_effect(self.current_image, features)
        elif style == "abstract":
            self.processed_image = self.art_filters.apply_abstract_art(self.current_image, features)
        elif style == "highlight":
            self.processed_image = self.art_filters.apply_feature_highlighting(self.current_image, features)
            
        # Display processed image
        if self.processed_image is not None:
            self.display_image(self.processed_image, self.processed_label)
            
    def save_image(self):
        if self.processed_image is None:
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            
    def display_image(self, image, label, max_size=400):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image while maintaining aspect ratio
        height, width = image_rgb.shape[:2]
        scale = min(max_size/width, max_size/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image_resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Convert to PhotoImage
        image_pil = Image.fromarray(image_resized)
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update label
        label.configure(image=image_tk)
        label.image = image_tk  # Keep a reference

def main():
    root = tk.Tk()
    app = FacialFeatureArtApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
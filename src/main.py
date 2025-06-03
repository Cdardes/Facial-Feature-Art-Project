import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
from face_detector import FaceDetector
from art_filters import ArtisticFilters
from image_generator import ImageGenerator

class FacialFeatureArtApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Feature Art Creator")
        
        # Initialize components
        self.detector = FaceDetector()
        self.art_filters = ArtisticFilters()
        try:
            self.image_generator = ImageGenerator()
            self.has_generator = True
        except ValueError as e:
            self.has_generator = False
            messagebox.showwarning("Warning", str(e))
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.filter_tab = ttk.Frame(self.notebook, padding="10")
        self.generator_tab = ttk.Frame(self.notebook, padding="10")
        
        self.notebook.add(self.filter_tab, text="Art Filters")
        self.notebook.add(self.generator_tab, text="Image Generator")
        
        # Setup filter tab
        self.setup_filter_tab()
        
        # Setup generator tab
        self.setup_generator_tab()
        
    def setup_filter_tab(self):
        # Create buttons
        ttk.Button(self.filter_tab, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=5, pady=5)
        
        # Create filter selection
        ttk.Label(self.filter_tab, text="Select Art Style:").grid(row=1, column=0, padx=5, pady=5)
        self.style_var = tk.StringVar(value="sketch")
        styles = [
            "sketch", "cartoon", "pop_art", "abstract", "highlight",
            "neon_glow", "mosaic", "watercolor", "comic_book", "double_exposure"
        ]
        self.style_combo = ttk.Combobox(self.filter_tab, textvariable=self.style_var, values=styles)
        self.style_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Create apply button
        ttk.Button(self.filter_tab, text="Apply Effect", command=self.apply_effect).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create save button
        ttk.Button(self.filter_tab, text="Save Image", command=self.save_image).grid(row=2, column=2, pady=10)
        
        # Create image display areas
        self.original_label = ttk.Label(self.filter_tab, text="Original Image")
        self.original_label.grid(row=3, column=0, padx=5, pady=5)
        
        self.processed_label = ttk.Label(self.filter_tab, text="Processed Image")
        self.processed_label.grid(row=3, column=1, padx=5, pady=5)
        
    def setup_generator_tab(self):
        if not self.has_generator:
            ttk.Label(self.generator_tab, text="Image generation is not available.\nPlease set up your Hugging Face API token.").grid(row=0, column=0, padx=5, pady=5)
            return
            
        # Create prompt input
        ttk.Label(self.generator_tab, text="Enter Prompt:").grid(row=0, column=0, padx=5, pady=5)
        self.prompt_text = scrolledtext.ScrolledText(self.generator_tab, width=40, height=4)
        self.prompt_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        # Create negative prompt input
        ttk.Label(self.generator_tab, text="Negative Prompt (Optional):").grid(row=2, column=0, padx=5, pady=5)
        self.negative_prompt_text = scrolledtext.ScrolledText(self.generator_tab, width=40, height=2)
        self.negative_prompt_text.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Create generate button
        ttk.Button(self.generator_tab, text="Generate Image", command=self.generate_image).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Create image display
        self.generated_label = ttk.Label(self.generator_tab, text="Generated Image")
        self.generated_label.grid(row=5, column=0, columnspan=2, padx=5, pady=5)
        
        # Create save button for generated image
        ttk.Button(self.generator_tab, text="Save Generated Image", command=self.save_generated_image).grid(row=6, column=0, columnspan=2, pady=10)
        
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
            messagebox.showerror("Error", "Please load an image first!")
            return
            
        # Detect facial features
        features = self.detector.detect_features(self.current_image)
        
        # Apply selected effect
        style = self.style_var.get()
        try:
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
            elif style == "neon_glow":
                self.processed_image = self.art_filters.apply_neon_glow(self.current_image, features)
            elif style == "mosaic":
                self.processed_image = self.art_filters.apply_mosaic_art(self.current_image, features)
            elif style == "watercolor":
                self.processed_image = self.art_filters.apply_watercolor(self.current_image)
            elif style == "comic_book":
                self.processed_image = self.art_filters.apply_comic_book(self.current_image, features)
            elif style == "double_exposure":
                self.processed_image = self.art_filters.apply_double_exposure(self.current_image, features)
            
            # Display processed image
            if self.processed_image is not None:
                self.display_image(self.processed_image, self.processed_label)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply effect: {str(e)}")
            
    def save_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "Please apply an effect first!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Success", "Image saved successfully!")
            
    def generate_image(self):
        if not self.has_generator:
            return
            
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showerror("Error", "Please enter a prompt!")
            return
            
        negative_prompt = self.negative_prompt_text.get("1.0", tk.END).strip()
        if not negative_prompt:
            negative_prompt = None
            
        try:
            # Show loading message
            self.generated_label.configure(text="Generating image... Please wait...")
            self.root.update()
            
            # Generate image
            self.current_image = self.image_generator.generate_from_prompt(prompt, negative_prompt)
            
            # Display generated image
            self.display_image(self.current_image, self.generated_label)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.generated_label.configure(text="Generated Image")
            
    def save_generated_image(self):
        if not hasattr(self, 'current_image') or self.current_image is None:
            messagebox.showerror("Error", "No image to save!")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.current_image)
            messagebox.showinfo("Success", "Image saved successfully!")
            
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
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

class PlantDiseaseDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Plant Disease Detection")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Load the YOLOv8 model
        try:
            self.model = YOLO("best.pt")  # Change to your model path
        except Exception as e:
            messagebox.showerror("Model Error", f"Error loading model: {str(e)}")
            self.model = None
        
        # Dictionary of disease descriptions
        self.disease_descriptions = {
            "healthy": "This plant appears to be healthy with no visible signs of disease.",
            "early_blight": "Early Blight is characterized by small, dark spots that grow larger with concentric rings, creating a 'bull's-eye' pattern. It primarily affects lower leaves first.",
            "late_blight": "Late Blight appears as dark, water-soaked spots on leaves that rapidly enlarge to form purple-brown lesions. White fungal growth may appear on the underside of leaves in humid conditions.",
            "bacterial_spot": "Bacterial Spot causes small, dark, water-soaked spots on leaves that later turn brown. The centers may fall out, giving a shot-hole appearance.",
            # Add more disease descriptions as needed
        }
        
        self.create_widgets()
        
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#4CAF50", height=60)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(header_frame, text="Automated Plant Disease Detection", 
                               font=("Arial", 18, "bold"), bg="#4CAF50", fg="white")
        header_label.pack(pady=10)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel for input
        left_panel = tk.Frame(main_frame, bg="#f0f0f0", width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=10)
        
        # Image selection
        select_frame = tk.LabelFrame(left_panel, text="Select Image", font=("Arial", 12), bg="#f0f0f0")
        select_frame.pack(fill=tk.X, pady=10)
        
        select_button = tk.Button(select_frame, text="Browse Image", font=("Arial", 10),
                                 command=self.select_image, bg="#008CBA", fg="white")
        select_button.pack(pady=10)
        
        # Image preview
        self.preview_frame = tk.LabelFrame(left_panel, text="Image Preview", font=("Arial", 12), bg="#f0f0f0")
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.preview_label = tk.Label(self.preview_frame, bg="#f0f0f0")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analyze button
        self.analyze_button = tk.Button(left_panel, text="Analyze Disease", font=("Arial", 12, "bold"),
                                       command=self.analyze_image, bg="#4CAF50", fg="white", state=tk.DISABLED)
        self.analyze_button.pack(fill=tk.X, pady=10)
        
        # Right panel for results
        self.right_panel = tk.Frame(main_frame, bg="#f0f0f0", width=400)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Initialize variables
        self.image_path = None
        self.tk_image = None
        self.result_window = None
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_path = file_path
            self.display_preview(file_path)
            self.analyze_button.config(state=tk.NORMAL)
    
    def display_preview(self, image_path):
        image = Image.open(image_path)
        image = image.resize((300, 300), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(image)
        self.preview_label.config(image=self.tk_image)
    
    def analyze_image(self):
        if not self.image_path or not self.model:
            messagebox.showerror("Error", "Please select an image and ensure model is loaded")
            return
        
        try:
            # Run inference with YOLOv8
            results = self.model(self.image_path)
            
            # Process results
            result_image = results[0].plot()  # Get the plotted image with detections
            
            # Get detected classes and their confidences
            detected_diseases = []
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    class_name = results[0].names[cls_id]
                    detected_diseases.append((class_name, conf))
            
            # If no diseases detected
            if not detected_diseases:
                detected_diseases = [("No disease detected", 0.0)]
            
            # Show results in a new window
            self.show_results(result_image, detected_diseases)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Error during image analysis: {str(e)}")
    
    def show_results(self, result_image, detected_diseases):
        # Create a new window for results if it doesn't exist
        if self.result_window is None or not self.root.winfo_exists():
            self.result_window = tk.Toplevel(self.root)
            self.result_window.title("Analysis Results")
            self.result_window.geometry("800x600")
            self.result_window.configure(bg="#f0f0f0")
        else:
            # Clear previous results
            for widget in self.result_window.winfo_children():
                widget.destroy()
        
        # Header
        header_frame = tk.Frame(self.result_window, bg="#4CAF50", height=60)
        header_frame.pack(fill=tk.X)
        
        header_label = tk.Label(header_frame, text="Disease Analysis Results", 
                               font=("Arial", 18, "bold"), bg="#4CAF50", fg="white")
        header_label.pack(pady=10)
        
        # Main content
        content_frame = tk.Frame(self.result_window, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - image with detections
        image_frame = tk.LabelFrame(content_frame, text="Detection Result", font=("Arial", 12), bg="#f0f0f0")
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Convert OpenCV image to PIL format
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        
        # Resize to fit
        result_pil = result_pil.resize((350, 350), Image.LANCZOS)
        result_tk = ImageTk.PhotoImage(result_pil)
        
        result_label = tk.Label(image_frame, image=result_tk, bg="#f0f0f0")
        result_label.image = result_tk  # Keep a reference
        result_label.pack(padx=10, pady=10)
        
        # Right side - results and description
        info_frame = tk.Frame(content_frame, bg="#f0f0f0")
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Results section
        results_frame = tk.LabelFrame(info_frame, text="Detection Results", font=("Arial", 12), bg="#f0f0f0")
        results_frame.pack(fill=tk.X, pady=10)
        
        for disease, confidence in detected_diseases:
            result_text = f"{disease.replace('_', ' ').title()}: {confidence:.2f}"
            tk.Label(results_frame, text=result_text, font=("Arial", 11), 
                    bg="#f0f0f0", anchor="w").pack(fill=tk.X, padx=10, pady=5)
        
        # Description section
        description_frame = tk.LabelFrame(info_frame, text="Disease Description", font=("Arial", 12), bg="#f0f0f0")
        description_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Get the disease with highest confidence
        if detected_diseases:
            primary_disease = detected_diseases[0][0]
            description = self.disease_descriptions.get(
                primary_disease, 
                f"No detailed information available for {primary_disease}."
            )
        else:
            description = "No diseases detected in this image."
        
        description_text = tk.Text(description_frame, wrap=tk.WORD, font=("Arial", 11), 
                                  bg="#f0f0f0", height=10)
        description_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        description_text.insert(tk.END, description)
        description_text.config(state=tk.DISABLED)
        
        # Close button
        close_button = tk.Button(self.result_window, text="Close", font=("Arial", 12),
                                command=self.result_window.destroy, bg="#f44336", fg="white")
        close_button.pack(pady=10)

def main():
    root = tk.Tk()
    app = PlantDiseaseDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
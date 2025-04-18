import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load trained CNN model
try:
    model = load_model('rice_cnn_model.h5')
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit()

# Label dictionary
labels_dict = {
    0: 'Arborio',
    1: 'Basmati',
    2: 'Ipsala',
    3: 'Jasmine',
    4: 'Karacadag'
}

# GUI Class
class RiceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Detector (CNN)")
        self.root.geometry("800x600")

        self.label = tk.Label(root, text="Select an image to detect rice type", font=("Arial", 14))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Image", command=self.select_image, font=("Arial", 12))
        self.select_button.pack(pady=10)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Rice Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        
        if not file_path:
            return

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Error", "Could not load the image!")
            return

        # Resize and preprocess for CNN
        cnn_input = cv2.resize(image, (224, 224))
        cnn_input = cnn_input / 255.0
        cnn_input = np.expand_dims(cnn_input, axis=0)

        # Display image
        display_image = cv2.resize(image, (660, 380))
        
        # Predict with CNN
        probs = model.predict(cnn_input)[0]
        max_prob = np.max(probs)
        prediction = np.argmax(probs)
        print(f"Prediction index: {prediction}, Confidence: {max_prob}")
        print(f"Probabilities: {probs}")

        # Result text
        if max_prob >= 0.5:  # Confidence থ্রেশহোল্ড
            predicted_value = labels_dict.get(prediction, "Unknown")
            result_text = f"Detected: {predicted_value} (Confidence: {max_prob:.2f})"
            cv2.rectangle(display_image, (0, 0), (660, 380), (0, 255, 0), 3)
        else:
            result_text = "No Rice Detected"

        # Convert image for Tkinter
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update GUI
        self.image_label.config(image=imgtk)
        self.image_label.image = imgtk
        self.result_label.config(text=result_text)

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = RiceDetectorApp(root)
    root.mainloop()
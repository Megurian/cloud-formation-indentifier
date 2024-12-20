import tkinter as tk
from tkinter import Label, Button, filedialog, Frame
from PIL import Image, ImageTk
from PIL import UnidentifiedImageError
from teachable_machine import TeachableMachine
import cv2 as cv
import numpy as np


#Disable AutoGraph Globally
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


""" Logging Functionality Start Here """

import logging

logging.basicConfig(filename="debug_log.txt", level=logging.DEBUG)
logging.debug("Application started")

# Log file paths
logging.debug(f"Model path: {resource_path('keras_model.h5')}")
logging.debug(f"Labels path: {resource_path('labels.txt')}")

""" Logging Functionality End Here """

# Function to load class names, descriptions, and indications from text files
def load_class_names(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def load_descriptions(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def load_weather_indications(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

# Initialize the model
model = TeachableMachine(
    model_path=resource_path("keras_model.h5"),
    labels_file_path=resource_path("labels.txt"),
    model_type="h5"
)

# Load data from files
class_names = load_class_names(resource_path("labels.txt"))
descriptions = load_descriptions(resource_path("description.txt"))
indications = load_weather_indications(resource_path("indicator.txt"))

# Function to classify an image
def classify_image(file_path):
    img = cv.imread(file_path)
    if img is None:
        clear_results()
        result_label.config(text=f"Error: Could not load image from path: {file_path}")
        return

    # Resize image for display
    try:
        pil_image = Image.open(file_path).resize((250, 250))
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label.config(image=tk_image)
        image_label.image = tk_image  # Keep a reference to avoid garbage collection
    except UnidentifiedImageError:
        result_label.config(text="Error: Unsupported or corrupted image file.")
        return
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")
        return

    # Classify the image
    try:
        result = model.classify_image(file_path)
        class_index = result["class_index"]
        class_name = result["class_name"]
        highest_class_name = result["highest_class_name"]
        class_confidence = result["class_confidence"]
        predictions = result["predictions"]
    except Exception as e:
        logging.error(f"Error during model classification: {str(e)}")
        result_label.config(text=f"Error during classification: {str(e)}")
        return

    # Display results
    if class_confidence >= 0.80:
        result_label.config(text=f"AI Prediction: {class_name}\nConfidence: {class_confidence * 100:.2f}%")
        description_label.config(text=f"\u2022 Description:\n{descriptions[class_index]}", font=("Arial", 12, "bold"))
        indicator_label.config(text=f"\u2022 Weather Indication:\n{indications[class_index]}", font=("Arial", 12, "bold"))
    else:
        max_confidence = max(predictions)
        max_index = np.argmax(predictions)  # Corrected this line
        if max_confidence >= 0.50:
            max_class_name = class_names[max_index]
            print(max_index)
            print(max_class_name)
            result_label.config(text=f"AI Prediction: {max_class_name}\nConfidence: {max_confidence * 100:.2f}%")
            description_label.config(text=f"\u2022 Description:\n{descriptions[max_index]}", font=("Arial", 12, "bold"))
            indicator_label.config(text=f"\u2022 Weather Indication:\n{indications[max_index]}", font=("Arial", 12, "bold"))
        else:
            clear_results()
            result_label.config(text="Prediction confidence is too low.\n\nTry uploading/capturing different picture")

# Function to upload an image and classify it
def upload_and_classify():
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if file_path:
        classify_image(file_path)
    else:
        clear_results()
        result_label.config(text="No file selected.")

# Function to start live camera feed
def start_camera_feed():
    global cap
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="Error: Camera could not be opened.")
        return

    live_feed_label.configure(width=250, height=250)
    set_image_empty()

    def update_feed():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            live_feed_label.imgtk = imgtk
            live_feed_label.configure(image=imgtk)
            live_feed_label.after(10, update_feed)
        if 'cap' in globals() and cap.isOpened():
            start_camera_button.config(state=tk.DISABLED)
    
    update_feed()

# Function to capture an image from the camera
def capture_from_camera():
    start_camera_button.config(state=tk.NORMAL)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            temp_path = "captured_image.jpg"
            cv.imwrite(temp_path, frame)
            classify_image(temp_path)
            remove_live_feed()
        else:
            clear_results()
            result_label.config(text="Error capturing image.")
    else:
        clear_results()
        result_label.config(text="Camera is not open.")

def remove_live_feed():
    if 'cap' in globals() and cap.isOpened():
        cap.release()
    live_feed_label.configure(image="", width=0, height=0)
    live_feed_label.imgtk = None

def set_image_empty():
    image_label.configure(image='')

def clear_results():
    result_label.configure(text='')
    description_label.configure(text='')
    indicator_label.configure(text='')

# Main UI setup
root = tk.Tk()
root.title("Ulap: Cloud Classification")
root.geometry("600x750")
root.configure(bg="#e8f4f8")

# Header frame
header_frame = Frame(root, bg="#0056a6", pady=10)
header_frame.pack(fill=tk.X)

header_label = Label(header_frame, text="Ulap: Cloud Formation Identification", font=("Arial", 20, "bold"), fg="white", bg="#0056a6")
header_label.pack()

# Buttons frame
button_frame = Frame(root, bg="#e8f4f8", pady=10)
button_frame.pack()

upload_button = Button(button_frame, text="Upload Image", command=upload_and_classify, font=("Arial", 14), bg="#0056a6", fg="white", width=15)
upload_button.grid(row=0, column=0, padx=10)

start_camera_button = Button(button_frame, text="Start Camera", command=start_camera_feed, font=("Arial", 14), bg="#0056a6", fg="white", width=15)
start_camera_button.grid(row=0, column=1, padx=10)

capture_button = Button(button_frame, text="Capture Image", command=capture_from_camera, font=("Arial", 14), bg="#0056a6", fg="white", width=15)
capture_button.grid(row=0, column=2, padx=10)

# Live feed label
live_feed_label = Label(root, bg="#dce3e8")
live_feed_label.pack(pady=10)

# Image display
image_label = Label(root, bg="#dce3e8", width=250, height=250)
image_label.pack(pady=10)

# Result and details frame
result_frame = Frame(root, bg="#e8f4f8", pady=10)
result_frame.pack(fill=tk.X)

result_label = Label(result_frame, text="Choose an option to classify a cloud image", font=("Arial", 12), fg="#0056a6", bg="#e8f4f8", wraplength=550, justify=tk.CENTER)
result_label.pack(pady=5)

description_label = Label(result_frame, text="", font=("Arial", 12, "bold"), fg="#333", bg="#e8f4f8", wraplength=550, justify=tk.LEFT)
description_label.pack(pady=5)

indicator_label = Label(result_frame, text="", font=("Arial", 12, "bold"), fg="#333", bg="#e8f4f8", wraplength=550, justify=tk.LEFT)
indicator_label.pack(pady=5)

# Run the app
root.mainloop()

# pip install ollama
import ollama
import cv2
import json
import matplotlib.pyplot as plt

# response 1
# Step 1: Ask Llama 3.2 Vision to locate the object
response1 = ollama.chat(
    model="llama3.2-vision",
    messages=[{
        "role": "user",
        "content": "Locate the red hat in this image and return only its bounding box in JSON format of {x: , y: , width: , height: } with un-normalised pixel values.",
        "images": ["red_hat.jpg"]  # Replace with your actual image path
    }]
)

print(response1)


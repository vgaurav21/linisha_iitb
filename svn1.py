import os
import subprocess
import threading
import matplotlib.pyplot as plt
import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate

# Ensure CPU usage
device = torch.device("cpu")

# Directory and file paths
HOME = os.getcwd()
CONFIG_PATH = os.path.join(HOME, "GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME")
SNAPS_DIR = "/home/gaurav/snaps"

# Validate file paths
assert os.path.isfile(CONFIG_PATH), "Config file not found!"
assert os.path.isfile(WEIGHTS_PATH), "Weights file not found!"
assert os.path.isdir(SNAPS_DIR), "Snaps directory not found!"

# Load the model
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
model = model.to(device)  # Move model to CPU

# Start frame capture
def start_frame_capture():
    print("Starting frame capture...")
    capture_cmd = (
        "rpicam-vid --timeout 0 --inline --framerate 30 -o - | "
        f"ffmpeg -i - -vf fps=1/2 {SNAPS_DIR}/frame_%04d.jpg"
    )
    subprocess.run(capture_cmd, shell=True)

# Process frames for object detection
def process_frames():
    try:
        while True:
            # Find the latest frame
            all_frames = sorted(os.listdir(SNAPS_DIR))
            latest_frame = os.path.join(SNAPS_DIR, all_frames[-1]) if all_frames else None

            if latest_frame and os.path.isfile(latest_frame):
                print(f"Processing {latest_frame}")

                # Load image
                image_source, image = load_image(latest_frame)

                # Generate predictions
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption="person",
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=device  # Ensure predictions happen on CPU
                )

                # Annotate the image
                annotated_frame = annotate(
                    image_source=image_source,
                    boxes=boxes,
                    logits=logits,
                    phrases=phrases
                )

                # Display the annotated image
                plt.imshow(annotated_frame)
                plt.axis('off')
                plt.show()

    except KeyboardInterrupt:
        print("Frame processing terminated by user.")

# Run frame capture and processing in parallel
try:
    # Start streaming and processing in separate threads
    capture_thread = threading.Thread(target=start_frame_capture, daemon=True)
    processing_thread = threading.Thread(target=process_frames, daemon=True)

    capture_thread.start()
    processing_thread.start()

    # Keep the main thread active
    capture_thread.join()
    processing_thread.join()

except KeyboardInterrupt:
    print("Script terminated by user.")

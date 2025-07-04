import streamlit as st
import torch
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
import sys

# Add yolov5 folder to system path
sys.path.append('yolov5')  # Path to yolov5 folder

# Load custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/yolov5.pt', source='local')

st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("🖼️ YOLO Object Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to numpy array
    image_np = np.array(image)

    # Inference
    results = model(image_np)

    # Draw results
    results.render()
    rendered_image = Image.fromarray(results.ims[0])

    # Display image with bounding boxes
    st.image(rendered_image, caption='Detected Objects', use_column_width=True)

    # Extract detection data
    detections = results.pandas().xyxy[0]

    if not detections.empty:
        # Show detection table
        st.subheader("📋 Detection Details")
        st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])

        # Show count of each detected object
        count_data = dict(Counter(detections['name']))
        count_df = pd.DataFrame(list(count_data.items()), columns=['Object', 'Count'])
        st.subheader("🔢 Object Counts")
        st.table(count_df)
    else:
        st.warning("No objects detected.")

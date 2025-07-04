import streamlit as st
import torch
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
import sys

# Load YOLOv5 model
sys.path.append('yolov5')  # path to yolov5 folder
model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Use yolov5s model

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

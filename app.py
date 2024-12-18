import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# Define paths for models
BINARY_MODEL_PATH = "models/gas_leak_3dcnn_final.h5"
THREE_CLASS_MODEL_PATH = "models/3Class_gas_leak_3dcnn_final.h5"

# Load pre-trained models
@st.cache_resource
def load_models():
    return (
        load_model(BINARY_MODEL_PATH),
        load_model(THREE_CLASS_MODEL_PATH),
    )

binary_model, three_class_model = load_models()

# Define preprocessing function
def preprocess_frame(frame, background_frames, background_window):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)

    background_frames.append(gray_frame)

    # Compute the background using the median of the deque
    if len(background_frames) < 210:
        temp_background = np.mean(background_frames, axis=0).astype(np.uint8)
        foreground = cv2.absdiff(gray_frame, temp_background)
    else:
    # Median background once buffer is full
        median_background = np.median(np.array(background_frames), axis=0).astype(np.uint8)
        foreground = cv2.absdiff(gray_frame, median_background)

                # Apply a threshold to the foreground
    _, foreground = cv2.threshold(foreground, 15, 255, cv2.THRESH_BINARY)
    normalized_frame = foreground / 255.0  # Normalize to [0, 1]
    return normalized_frame

# Real-time classification and video processing
def classify_video(video_path, model, frame_shape, num_frames, class_labels):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open video.")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    background_frames = deque(maxlen=210)
    frames = []
    predictions = []

    # Font colors for binary and three-class
    binary_colors = {"No Leak": (255, 255, 255), "Leak": (0, 255, 0)}  # White and Green
    three_class_colors = {"Small": (0, 255, 255), "Medium": (0, 165, 255), "Large": (0, 0, 255)}  # Yellow, Orange, Red

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame, background_frames, background_window=210)
        frames.append(processed_frame)

        if len(frames) == num_frames:
            segment = np.array(frames).reshape(1, num_frames, *frame_shape, 1)
            prediction = model.predict(segment)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = np.max(prediction)
            predictions.append((predicted_class, confidence))
            frames.pop(0)

    # Overlay predictions on frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(predictions):
            break

        pred_class, conf = predictions[frame_idx]
        label = f"{class_labels[pred_class]} ({conf:.2f})"

        # Select color based on classification type
        if len(class_labels) == 2:  # Binary classification
            color = binary_colors[class_labels[pred_class]]
        else:  # Three-class classification
            color = three_class_colors[class_labels[pred_class]]

        cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    return output_video_path

# Streamlit app
st.title("Real-Time Gas Leak Classification")
st.sidebar.header("Settings")
classification_type = st.sidebar.radio("Choose Classification Type", ["Binary Classification", "Three-Class Classification"])
uploaded_video = st.file_uploader("Upload a video file (MP4 format)", type=["mp4"])

if uploaded_video:
    st.video(uploaded_video)

    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Select the appropriate model and class labels
    if classification_type == "Binary Classification":
        model = binary_model
        class_labels = ["No Leak", "Leak"]
    else:
        model = three_class_model
        class_labels = ["Small", "Medium", "Large"]

    output_video_path = classify_video(temp_video_path, model, (240, 320), 15, class_labels)

    if output_video_path:
        st.success("Processing complete!")
        st.video(output_video_path)
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

import streamlit as st
import cv2
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Pre-trained Models
@st.cache_resource
def load_binary_model():
    return load_model("binary_classification_model.h5")  # Replace with your binary classification model path

@st.cache_resource
def load_three_class_model():
    return load_model("three_class_classification_model.h5")  # Replace with your three-class model path

# Preprocess Video with MOG2 Background Subtraction
def preprocess_video_mog2(video_path, frame_shape=(15, 240, 320, 1)):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Initialize MOG2 background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and resize to model's input shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (frame_shape[2], frame_shape[1]))

        # Apply MOG2 background subtraction
        foreground = bg_subtractor.apply(resized_frame)

        # Threshold the foreground
        _, foreground = cv2.threshold(foreground, 5, 255, cv2.THRESH_BINARY)

        # Normalize and append
        normalized_frame = foreground / 255.0
        frames.append(normalized_frame)

    cap.release()

    # Create video segments of required shape
    frame_count = len(frames)
    segments = []

    for i in range(0, frame_count - frame_shape[0] + 1, frame_shape[0]):
        segment = frames[i:i + frame_shape[0]]
        segment = np.array(segment).reshape(frame_shape)
        segments.append(segment)

    return np.array(segments)

# Predict Class Labels
def predict_binary(model, video_segments):
    predictions = model.predict(video_segments)
    return np.argmax(predictions, axis=1)  # 0 for no_leak, 1 for leak

def predict_three_class(model, video_segments):
    predictions = model.predict(video_segments)
    return np.argmax(predictions, axis=1)  # 0 for small, 1 for medium, 2 for large

# Streamlit Web App
st.title("Video Classification Web App with MOG2 Preprocessing")
st.write("Upload a video to classify using the trained models.")

# Classification Mode Selection
classification_mode = st.selectbox(
    "Select Classification Mode",
    ("Binary Classification (Leak/No Leak)", "Three-Class Classification (Small/Medium/Large)")
)

# File Uploader
uploaded_video = st.file_uploader("Upload a video file (MP4)", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    st.video(temp_video_path)
    st.write("Video uploaded successfully! Processing...")

    # Preprocess Video with MOG2
    frame_shape = (15, 240, 320, 1)
    video_segments = preprocess_video_mog2(temp_video_path, frame_shape)

    if classification_mode == "Binary Classification (Leak/No Leak)":
        model = load_binary_model()
        predictions = predict_binary(model, video_segments)
        class_counts = {"No Leak": np.sum(predictions == 0), "Leak": np.sum(predictions == 1)}

        st.write("**Binary Classification Results:**")
        st.write(class_counts)

    elif classification_mode == "Three-Class Classification (Small/Medium/Large)":
        model = load_three_class_model()
        predictions = predict_three_class(model, video_segments)
        class_counts = {
            "Small Leak": np.sum(predictions == 0),
            "Medium Leak": np.sum(predictions == 1),
            "Large Leak": np.sum(predictions == 2),
        }

        st.write("**Three-Class Classification Results:**")
        st.write(class_counts)

    # Display Predictions Summary
    st.write("**Processed Segments:**")
    st.write(f"Total Segments Processed: {len(video_segments)}")
else:
    st.write("Please upload a video file to proceed.")

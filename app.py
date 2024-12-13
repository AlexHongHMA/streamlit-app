import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import tempfile
import os

# Define categories for classification
BINARY_CATEGORIES = ['No Leak', 'Leak']
THREE_CLASS_CATEGORIES = ['Small', 'Medium', 'Large']

# Define paths for models
BINARY_MODEL_PATH = "models/gas_leak_3dcnn_final.h5"
THREE_CLASS_MODEL_PATH = "models/3Class_gas_leak_3dcnn_final.h5"

# Load models
@st.cache_resource
def load_binary_model():
    model = tf.keras.models.load_model(BINARY_MODEL_PATH)
    return model

@st.cache_resource
def load_three_class_model():
    model = tf.keras.models.load_model(THREE_CLASS_MODEL_PATH)
    return model

def preprocess_frame(frame, bg_subtractor, gaussian_kernel_size=(7, 7)):
    """
    Preprocess a single frame for the model:
    1. Convert to grayscale
    2. Apply Gaussian blur
    3. Apply MOG2 background subtraction
    4. Normalize and reshape
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, gaussian_kernel_size, 0)
    foreground = bg_subtractor.apply(gray_frame)
    _, foreground = cv2.threshold(foreground, 5, 255, cv2.THRESH_BINARY)
    normalized_frame = foreground / 255.0
    return normalized_frame.reshape(240, 320, 1)

def process_video_with_labels(video_path, output_path, model, categories, frame_size=(240, 320, 1), sliding_window_size=15):
    """
    Process video frame by frame, classify with the model, and save a new video with predictions overlaid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=False)
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the current frame
        preprocessed_frame = preprocess_frame(frame, bg_subtractor)
        frames_buffer.append(preprocessed_frame)

        # When the buffer has enough frames, make a prediction
        if len(frames_buffer) == sliding_window_size:
            input_data = np.expand_dims(frames_buffer, axis=0)  # Add batch dimension
            prediction = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(prediction)
            label = categories[predicted_class]

            # Draw the label on the frame
            cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Slide the window
            frames_buffer.pop(0)

        # Write the frame with the label to the output video
        out.write(frame)

    cap.release()
    out.release()

# Streamlit App
def main():
    st.title("Gas Leak Classification (Real-Time with Labels)")
    st.write("Upload a video to classify gas leaks in real time with predictions displayed on the video.")

    # Classification mode selection
    classification_mode = st.selectbox(
        "Select Classification Mode",
        ("Binary Classification (Leak/No Leak)", "Three-Class Classification (Small/Medium/Large)")
    )

    # File uploader
    uploaded_video = st.file_uploader("Upload a video file (MP4, AVI)", type=["mp4", "avi"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name

        st.video(temp_video_path)
        st.write("Processing video...")

        if classification_mode == "Binary Classification (Leak/No Leak)":
            model = load_binary_model()
            categories = BINARY_CATEGORIES
        else:
            model = load_three_class_model()
            categories = THREE_CLASS_CATEGORIES

        # Save processed video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_output:
            output_video_path = temp_output.name

        # Process the video and add predictions
        process_video_with_labels(temp_video_path, output_video_path, model, categories)

        st.write("Processed video with predictions:")
        st.video(output_video_path)

if __name__ == "__main__":
    main()

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import io
import cv2

st.set_page_config(page_title="Pinworm Egg Detection", layout="wide")
st.title("Pinworm Egg Detection (YOLO)")

st.sidebar.header("Navigation")
model_path = "best.pt"
device_option = "cpu"
confidence_threshold = 0.5
page = st.sidebar.radio("Go to", options=["Home", "App"], index=0)
@st.cache_resource
def load_model(path, device):
    return YOLO(path) if device == "cpu" else YOLO(path)  # ultralytics handles device at predict time
if page == "Home":
    st.header("Welcome to the Pinworm Egg Detection App")
    st.markdown("""
    This application uses a YOLO model to detect pinworm eggs in images.
    
    **Instructions:**
    1. Navigate to the 'App' section using the sidebar.
    2. Upload an image containing potential pinworm eggs.
    3. Click 'Run detection' to see the results.
    
    **Model Details:**
    - Model Path: `best.pt`
    - Confidence Threshold: 0.5
    - Device: CPU
    
    **Note:** Ensure that the uploaded images are clear for optimal detection performance.
    """)
    st.image("https://example.com/pinworm_eggs_image.jpg", caption="Pinworm Eggs Example", use_column_width=True)

elif page == "App":
    try:
        model = load_model(model_path, device_option)
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()

    # Allow user to either upload a file or take a picture with the webcam
    st.info("You can either upload an image or take a photo with your camera.")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    with col2:
        camera_image = st.camera_input("Take a picture")

    # Use camera image if provided, otherwise fall back to uploaded file
    image_source = camera_image if camera_image is not None else uploaded_file
    run_button = st.button("Run detection")

    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption="Input image")

        if run_button:
            with st.spinner("Running inference..."):
                # Save uploaded image to a temporary file for YOLO
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    image.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    results = model.predict(
                        source=tmp_path,
                        device=0 if device_option == "cuda" else "cpu",
                        imgsz=1024,
                        conf=confidence_threshold,
                        save=False,
                        show=False
                    )
                    if len(results) == 0:
                        st.warning("No results returned by the model.")
                    else:
                        r = results[0]
                        # Annotated image as numpy array
                        annotated = r.plot()  # returns RGB ndarray
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption="Annotated image")

                        # Number of detections
                        num_detections = len(r.boxes)
                        st.success(f"Total objects detected: {num_detections}")
                except Exception as e:
                    st.error(f"Inference failed: {e}")
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
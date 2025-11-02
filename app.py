import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import io

st.set_page_config(page_title="Pinworm Egg Detection", layout="wide")
st.title("Pinworm Egg Detection (YOLO)")

st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model path", value="best.pt")
device_option = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)

@st.cache_resource
def load_model(path, device):
    return YOLO(path) if device == "cpu" else YOLO(path)  # ultralytics handles device at predict time

try:
    model = load_model(model_path, device_option)
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
run_button = st.button("Run detection")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input image", use_column_width=True)

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
                    st.image(annotated, caption="Annotated image", use_column_width=True)

                    # Number of detections
                    num_detections = len(r.boxes)
                    st.success(f"Total objects detected: {num_detections}")

                    # Show details table if any detections
                    if num_detections > 0:
                        try:
                            boxes = r.boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2
                            scores = r.boxes.conf.cpu().numpy()
                            classes = r.boxes.cls.cpu().numpy().astype(int)
                            rows = []
                            for i, (b, s, c) in enumerate(zip(boxes, scores, classes), start=1):
                                rows.append({
                                    "id": i,
                                    "class": int(c),
                                    "confidence": float(s),
                                    "x1": float(b[0]),
                                    "y1": float(b[1]),
                                    "x2": float(b[2]),
                                    "y2": float(b[3])
                                })
                            st.table(rows)
                        except Exception:
                            st.info("Detections present but couldn't extract box details.")
            except Exception as e:
                st.error(f"Inference failed: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
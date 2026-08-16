import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2

# Page Configuration


st.set_page_config(
    page_title="Tennis Object Detection",
    page_icon="🎾",
    layout="wide"
)

=
# Load Model


MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()



# Title


st.title("🎾 Tennis Object Detection")
st.write(
    "Upload an image or video and the YOLO11 model will detect "
    "players, tennis balls, the net, and the court."
)

# Select Input Type

input_type = st.radio(
    "Choose input type:",
    ["Image", "Video"],
    horizontal=True
)

# IMAGE DETECTION

if input_type == "Image":

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        if st.button("🔍 Detect Objects"):

            with st.spinner("Running detection..."):

                results = model.predict(
                    source=image,
                    imgsz=960,
                    conf=0.25
                )

                result_image = results[0].plot()

            st.subheader("Detection Result")

            st.image(
                result_image,
                channels="BGR",
                use_container_width=True
            )

            st.success("Detection completed!")

# VIDEO DETECTION

else:

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        # Save uploaded video temporarily
        input_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_video.write(uploaded_video.read())
        input_video.close()

        st.video(input_video.name)

        if st.button("🎥 Run Detection"):

            with st.spinner(
                "Processing video... This may take some time."
            ):

                # Open input video
                cap = cv2.VideoCapture(input_video.name)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Output video
                output_video = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                )

                output_video.close()

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                out = cv2.VideoWriter(
                    output_video.name,
                    fourcc,
                    fps,
                    (width, height)
                )

                # Process frames
                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.predict(
                        source=frame,
                        imgsz=960,
                        conf=0.25,
                        verbose=False
                    )

                    annotated_frame = results[0].plot()

                    out.write(annotated_frame)

                # Release resources
                cap.release()
                out.release()

            st.success("Video detection completed!")

            st.subheader("Detection Result")

            st.video(output_video.name)

            # Download button
            with open(output_video.name, "rb") as video_file:

                st.download_button(
                    label="⬇️ Download Result Video",
                    data=video_file,
                    file_name="tennis_detection_result.mp4",
                    mime="video/mp4"
                )

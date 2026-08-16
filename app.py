import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Tennis Object Detection",
    page_icon="🎾",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


model = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🎾 Tennis Object Detection")
st.write(
    "Upload an image or video and the YOLO11 model will detect "
    "players and tennis balls."
)


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05
)

img_size = st.sidebar.selectbox(
    "Image Size",
    [640, 960, 1280],
    index=2
)


# ============================================================
# INPUT TYPE
# ============================================================

input_type = st.radio(
    "Choose input type:",
    ["Image", "Video"],
    horizontal=True
)


# ============================================================
# IMAGE DETECTION
# ============================================================

if input_type == "Image":

    uploaded_file = st.file_uploader(
        "Upload a tennis image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🔍 Detect Objects"):

            with st.spinner("Running detection..."):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=img_size,
                    verbose=False
                )

                result = results[0]

                # YOLO plot returns BGR
                annotated_image = result.plot()

                # Convert BGR → RGB
                annotated_image = cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                )

            st.subheader("Detection Result")

            st.image(
                annotated_image,
                use_container_width=True
            )

            # ----------------------------------------
            # Detection Summary
            # ----------------------------------------

            st.subheader("Detection Summary")

            if len(result.boxes) == 0:

                st.warning(
                    "No objects were detected."
                )

            else:

                detections = []

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    conf = float(box.conf[0])

                    detections.append(
                        {
                            "Class": class_name,
                            "Confidence": round(conf, 3)
                        }
                    )

                st.dataframe(
                    detections,
                    use_container_width=True
                )

                st.success(
                    f"{len(result.boxes)} object(s) detected."
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a tennis video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:

        if st.button("🎬 Detect Objects in Video"):

            # ----------------------------------------
            # Save uploaded video temporarily
            # ----------------------------------------

            input_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(
                    uploaded_video.name
                )[1]
            )

            input_video.write(
                uploaded_video.read()
            )

            input_video.close()

            # ----------------------------------------
            # Create output video path
            # ----------------------------------------

            output_video = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_video.close()

            # ----------------------------------------
            # Open input video
            # ----------------------------------------

            cap = cv2.VideoCapture(
                input_video.name
            )

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if fps <= 0:
                fps = 30

            # ----------------------------------------
            # Video writer
            # ----------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_video.name,
                fourcc,
                fps,
                (width, height)
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            progress_bar = st.progress(0)

            frame_count = 0

            # ----------------------------------------
            # Process video frame by frame
            # ----------------------------------------

            with st.spinner(
                "Processing video..."
            ):

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=img_size,
                        verbose=False
                    )

                    result = results[0]

                    annotated_frame = result.plot()

                    writer.write(
                        annotated_frame
                    )

                    frame_count += 1

                    if total_frames > 0:

                        progress = (
                            frame_count /
                            total_frames
                        )

                        progress_bar.progress(
                            min(progress, 1.0)
                        )

            # ----------------------------------------
            # Release resources
            # ----------------------------------------
            cap.release()
            writer.release()

            progress_bar.empty()

            st.success(
                "Video processing completed!"
            )

            # Display output video
     

            st.subheader(
                "Detection Result"
            )

            with open(
                output_video.name,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )

            st.video(video_bytes)

            # ----------------------------------------
            # Download button
            # ----------------------------------------

            st.download_button(
                label="⬇️ Download Result Video",
                data=video_bytes,
                file_name="tennis_detection_result.mp4",
                mime="video/mp4"
            )

            # ----------------------------------------
            # Cleanup
            # ----------------------------------------

            try:
                os.remove(
                    input_video.name
                )

                os.remove(
                    output_video.name
                )

            except Exception:
                pass

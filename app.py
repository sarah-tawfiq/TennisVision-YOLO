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
    "Upload an image or video and the YOLO11 model "
    "will detect tennis players and tennis balls."
)


# ============================================================
# SIDEBAR
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
    "Inference Image Size",
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

                # YOLO returns BGR image
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

            # ------------------------------------------------
            # Detection Summary
            # ------------------------------------------------

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
        type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:

        st.subheader("Uploaded Video")

        # ----------------------------------------------------
        # Show ORIGINAL video inside the website
        # ----------------------------------------------------

        original_video_bytes = uploaded_video.getvalue()

        st.video(
            original_video_bytes
        )

        if st.button("🎬 Run Detection on Video"):

            # ------------------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------------------

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                original_video_bytes
            )

            input_file.close()

            input_path = input_file.name

            # ------------------------------------------------
            # Open video
            # ------------------------------------------------

            cap = cv2.VideoCapture(
                input_path
            )

            if not cap.isOpened():

                st.error(
                    "Could not open the uploaded video."
                )

                os.remove(input_path)

                st.stop()

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

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            # ------------------------------------------------
            # Temporary output file
            # ------------------------------------------------

            output_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            output_file.close()

            output_path = output_file.name

            # ------------------------------------------------
            # H264 VideoWriter
            # ------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"avc1"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )

            # ------------------------------------------------
            # Check VideoWriter
            # ------------------------------------------------

            if not writer.isOpened():

                # Fallback codec
                fourcc = cv2.VideoWriter_fourcc(
                    *"mp4v"
                )

                writer = cv2.VideoWriter(
                    output_path,
                    fourcc,
                    fps,
                    (width, height)
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_count = 0

            # ------------------------------------------------
            # Process video frame-by-frame
            # ------------------------------------------------

            with st.spinner(
                "Running YOLO detection on video..."
            ):

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    # ----------------------------------------
                    # YOLO prediction
                    # ----------------------------------------

                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=img_size,
                        verbose=False
                    )

                    result = results[0]

                    # ----------------------------------------
                    # Draw bounding boxes
                    # ----------------------------------------

                    annotated_frame = result.plot()

                    # ----------------------------------------
                    # Write processed frame
                    # ----------------------------------------

                    writer.write(
                        annotated_frame
                    )

                    frame_count += 1

                    # ----------------------------------------
                    # Update progress
                    # ----------------------------------------

                    if total_frames > 0:

                        progress = (
                            frame_count /
                            total_frames
                        )

                        progress_bar.progress(
                            min(progress, 1.0)
                        )

                        status_text.write(
                            f"Processing frame "
                            f"{frame_count} / "
                            f"{total_frames}"
                        )

            # ------------------------------------------------
            # Release resources
            # ------------------------------------------------

            cap.release()
            writer.release()

            progress_bar.progress(1.0)

            status_text.success(
                "Video detection completed!"
            )

            # ------------------------------------------------
            # Read processed video
            # ------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as f:

                processed_video = f.read()

            # ------------------------------------------------
            # SHOW RESULT INSIDE WEBSITE
            # ------------------------------------------------

            st.subheader(
                "🎾 Detection Result"
            )

            st.video(
                processed_video
            )

            # ------------------------------------------------
            # Download button
            # ------------------------------------------------

            st.download_button(
                label="⬇️ Download Detection Video",
                data=processed_video,
                file_name="tennis_detection_result.mp4",
                mime="video/mp4"
            )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            try:

                os.remove(
                    input_path
                )

                os.remove(
                    output_path
                )

            except Exception:
                pass

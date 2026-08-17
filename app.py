import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import subprocess
import imageio_ffmpeg


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

                # YOLO returns BGR
                annotated_image = result.plot()

                # BGR → RGB
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

        # ----------------------------------------------------
        # Show original video
        # ----------------------------------------------------

        original_video_bytes = uploaded_video.getvalue()

        st.subheader("Original Video")

        st.video(
            original_video_bytes
        )

        if st.button("🎬 Run Detection on Video"):

            # =================================================
            # SAVE INPUT VIDEO
            # =================================================

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            input_file.write(
                original_video_bytes
            )

            input_file.close()

            input_path = input_file.name


            # =================================================
            # OPEN INPUT VIDEO
            # =================================================

            cap = cv2.VideoCapture(
                input_path
            )

            if not cap.isOpened():

                st.error(
                    "Could not open the uploaded video."
                )

                os.remove(input_path)
                st.stop()


            # =================================================
            # VIDEO INFORMATION
            # =================================================

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
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


            # =================================================
            # RAW OUTPUT VIDEO
            # =================================================

            raw_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            raw_output.close()

            raw_output_path = raw_output.name


            # =================================================
            # VIDEO WRITER
            # =================================================

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                raw_output_path,
                fourcc,
                fps,
                (width, height)
            )

            if not writer.isOpened():

                st.error(
                    "Could not create output video."
                )

                cap.release()

                os.remove(input_path)
                os.remove(raw_output_path)

                st.stop()


            # =================================================
            # PROGRESS BAR
            # =================================================

            progress_bar = st.progress(0)

            status_text = st.empty()

            frame_count = 0


            # =================================================
            # YOLO DETECTION
            # =================================================

            with st.spinner(
                "Running YOLO detection on video..."
            ):

                while True:

                    ret, frame = cap.read()

                    if not ret:
                        break


                    # -----------------------------------------
                    # YOLO prediction
                    # -----------------------------------------

                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=img_size,
                        verbose=False
                    )

                    result = results[0]


                    # -----------------------------------------
                    # Draw bounding boxes
                    # -----------------------------------------

                    annotated_frame = result.plot()


                    # -----------------------------------------
                    # Write frame
                    # -----------------------------------------

                    writer.write(
                        annotated_frame
                    )


                    # -----------------------------------------
                    # Update progress
                    # -----------------------------------------

                    frame_count += 1

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


            # =================================================
            # RELEASE VIDEO RESOURCES
            # =================================================

            cap.release()
            writer.release()

            progress_bar.progress(1.0)

            status_text.success(
                "YOLO detection completed!"
            )


            # =================================================
            # CONVERT VIDEO TO WEB-FRIENDLY H.264
            # =================================================

            st.info(
                "Converting video to a web-compatible format..."
            )

            final_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            final_output.close()

            final_output_path = final_output.name


            # ------------------------------------------------
            # Get FFmpeg executable
            # ------------------------------------------------

            ffmpeg_path = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )


            # ------------------------------------------------
            # FFmpeg command
            # ------------------------------------------------

            command = [
                ffmpeg_path,

                "-y",

                "-i",
                raw_output_path,

                # H.264 codec
                "-c:v",
                "libx264",

                # Browser-compatible pixel format
                "-pix_fmt",
                "yuv420p",

                # Optimize MP4 for web streaming
                "-movflags",
                "+faststart",

                # Keep FPS
                "-r",
                str(fps),

                final_output_path
            ]


            # ------------------------------------------------
            # Run FFmpeg
            # ------------------------------------------------

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )


            # =================================================
            # CHECK CONVERSION
            # =================================================

            if process.returncode != 0:

                st.error(
                    "Video conversion failed."
                )

                st.code(
                    process.stderr.decode(
                        errors="ignore"
                    )
                )

                os.remove(input_path)
                os.remove(raw_output_path)

                st.stop()


            # =================================================
            # READ FINAL VIDEO
            # =================================================

            with open(
                final_output_path,
                "rb"
            ) as f:

                processed_video = f.read()


            # =================================================
            # SHOW DETECTION RESULT INSIDE WEBSITE
            # =================================================

            st.subheader(
                "🎾 Detection Result"
            )

            st.video(
                processed_video
            )


            # =================================================
            # DOWNLOAD RESULT
            # =================================================

            st.download_button(
                label="⬇️ Download Detection Video",

                data=processed_video,

                file_name="tennis_detection_result.mp4",

                mime="video/mp4"
            )


            # =================================================
            # CLEANUP
            # =================================================

            try:

                os.remove(
                    input_path
                )

                os.remove(
                    raw_output_path
                )

                os.remove(
                    final_output_path
                )

            except Exception:
                pass

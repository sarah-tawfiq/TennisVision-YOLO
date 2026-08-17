import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import cv2
import subprocess
import imageio_ffmpeg
import base64
import streamlit.components.v1 as components


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
# VIDEO DISPLAY FUNCTION
# ============================================================

def display_video(video_bytes, max_height="70vh"):

    video_base64 = base64.b64encode(
        video_bytes
    ).decode("utf-8")

    video_html = f"""
    <div style="
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        overflow: hidden;
    ">

        <video
            controls
            playsinline
            style="
                max-width: 100%;
                max-height: {max_height};
                width: auto;
                height: auto;
                object-fit: contain;
                display: block;
            "
        >

            <source
                src="data:video/mp4;base64,{video_base64}"
                type="video/mp4"
            >

            Your browser does not support the video tag.

        </video>

    </div>
    """

    components.html(
        video_html,
        height=650,
        scrolling=False
    )


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
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = Image.open(
            uploaded_file
        )


        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        st.subheader(
            "🖼️ Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )


        # ----------------------------------------------------
        # Detection button
        # ----------------------------------------------------

        if st.button(
            "🔍 Detect Objects"
        ):

            with st.spinner(
                "Running YOLO detection..."
            ):

                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=img_size,
                    verbose=False
                )

                result = results[0]


                # --------------------------------------------
                # Draw detections
                # --------------------------------------------

                annotated_image = result.plot()


                # --------------------------------------------
                # BGR → RGB
                # --------------------------------------------

                annotated_image = cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                )


            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.subheader(
                "🎾 Detection Result"
            )

            st.image(
                annotated_image,
                use_container_width=True
            )


            # ------------------------------------------------
            # Detection summary
            # ------------------------------------------------

            st.subheader(
                "📊 Detection Summary"
            )


            if len(result.boxes) == 0:

                st.warning(
                    "No objects were detected."
                )

            else:

                detections = []


                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        result.names[class_id]
                    )

                    conf = float(
                        box.conf[0]
                    )


                    detections.append(
                        {
                            "Class": class_name,
                            "Confidence": round(
                                conf,
                                3
                            )
                        }
                    )


                st.dataframe(
                    detections,
                    use_container_width=True
                )


                st.success(
                    f"{len(result.boxes)} "
                    f"object(s) detected."
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload a tennis video",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ]
    )


    if uploaded_video is not None:

        # ====================================================
        # READ ORIGINAL VIDEO
        # ====================================================

        original_video_bytes = (
            uploaded_video.getvalue()
        )


        # ====================================================
        # SHOW ORIGINAL VIDEO
        # ====================================================

        st.subheader(
            "🎥 Uploaded Video"
        )


        display_video(
            original_video_bytes
        )


        # ====================================================
        # RUN DETECTION
        # ====================================================

        if st.button(
            "🎬 Run Detection on Video"
        ):

            # =================================================
            # ORIGINAL FILE EXTENSION
            # =================================================

            original_extension = os.path.splitext(
                uploaded_video.name
            )[1]


            if original_extension == "":
                original_extension = ".mp4"


            # =================================================
            # SAVE INPUT VIDEO
            # =================================================

            input_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=original_extension
            )


            input_file.write(
                original_video_bytes
            )


            input_file.close()


            input_path = (
                input_file.name
            )


            # =================================================
            # OPEN VIDEO
            # =================================================

            cap = cv2.VideoCapture(
                input_path
            )


            if not cap.isOpened():

                st.error(
                    "Could not open the uploaded video."
                )

                os.remove(
                    input_path
                )

                st.stop()


            # =================================================
            # GET ORIGINAL VIDEO INFORMATION
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
            # VIDEO INFORMATION
            # =================================================

            st.info(
                f"Original video: "
                f"{width} × {height} | "
                f"FPS: {fps:.2f} | "
                f"Frames: {total_frames}"
            )


            # =================================================
            # CREATE RAW OUTPUT
            # =================================================

            raw_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )


            raw_output.close()


            raw_output_path = (
                raw_output.name
            )


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


                try:
                    os.remove(input_path)
                    os.remove(raw_output_path)
                except Exception:
                    pass


                st.stop()


            # =================================================
            # PROGRESS BAR
            # =================================================

            progress_bar = st.progress(
                0
            )


            status_text = st.empty()


            frame_count = 0


            # =================================================
            # PROCESS VIDEO
            # =================================================

            with st.spinner(
                "Running YOLO detection..."
            ):

                while True:

                    # -----------------------------------------
                    # Read frame
                    # -----------------------------------------

                    ret, frame = cap.read()


                    if not ret:
                        break


                    # -----------------------------------------
                    # YOLO
                    # -----------------------------------------

                    results = model.predict(
                        source=frame,
                        conf=confidence,
                        imgsz=img_size,
                        verbose=False
                    )


                    result = results[0]


                    # -----------------------------------------
                    # Draw detections
                    # -----------------------------------------

                    annotated_frame = (
                        result.plot()
                    )


                    # -----------------------------------------
                    # Preserve ORIGINAL dimensions
                    # -----------------------------------------

                    if (
                        annotated_frame.shape[1]
                        != width
                        or
                        annotated_frame.shape[0]
                        != height
                    ):

                        annotated_frame = cv2.resize(
                            annotated_frame,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )


                    # -----------------------------------------
                    # Write frame
                    # -----------------------------------------

                    writer.write(
                        annotated_frame
                    )


                    # -----------------------------------------
                    # Progress
                    # -----------------------------------------

                    frame_count += 1


                    if total_frames > 0:

                        progress = (
                            frame_count
                            / total_frames
                        )


                        progress_bar.progress(
                            min(
                                progress,
                                1.0
                            )
                        )


                        status_text.write(
                            f"Processing frame "
                            f"{frame_count} / "
                            f"{total_frames}"
                        )


            # =================================================
            # RELEASE
            # =================================================

            cap.release()

            writer.release()


            progress_bar.progress(
                1.0
            )


            status_text.success(
                "YOLO detection completed!"
            )


            # =================================================
            # FFMPEG CONVERSION
            # =================================================

            st.info(
                "Preparing detection video "
                "for web playback..."
            )


            final_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )


            final_output.close()


            final_output_path = (
                final_output.name
            )


            # =================================================
            # GET FFMPEG
            # =================================================

            ffmpeg_path = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )


            # =================================================
            # FFMPEG COMMAND
            # =================================================

            command = [

                ffmpeg_path,

                "-y",

                "-i",
                raw_output_path,

                # --------------------------------------------
                # Keep original dimensions
                # --------------------------------------------

                "-vf",
                (
                    "scale="
                    "trunc(iw/2)*2:"
                    "trunc(ih/2)*2"
                ),

                # --------------------------------------------
                # H264
                # --------------------------------------------

                "-c:v",
                "libx264",

                # --------------------------------------------
                # Browser compatible
                # --------------------------------------------

                "-pix_fmt",
                "yuv420p",

                # --------------------------------------------
                # Web streaming
                # --------------------------------------------

                "-movflags",
                "+faststart",

                # --------------------------------------------
                # FPS
                # --------------------------------------------

                "-r",
                str(fps),

                final_output_path
            ]


            # =================================================
            # RUN FFMPEG
            # =================================================

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )


            # =================================================
            # CHECK FFMPEG
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


                st.stop()


            # =================================================
            # READ PROCESSED VIDEO
            # =================================================

            with open(
                final_output_path,
                "rb"
            ) as f:

                processed_video = (
                    f.read()
                )


            # =================================================
            # DETECTION RESULT
            # =================================================

            st.subheader(
                "🎾 Detection Result"
            )


            # =================================================
            # SHOW ORIGINAL VIDEO AGAIN
            # =================================================

            st.markdown(
                "### 🎥 Original Video"
            )


            display_video(
                original_video_bytes
            )


            # =================================================
            # SHOW DETECTION VIDEO
            # =================================================

            st.markdown(
                "### 🎾 Detection Video"
            )


            display_video(
                processed_video
            )


            # =================================================
            # DOWNLOAD
            # =================================================

            st.download_button(
                label=(
                    "⬇️ Download Detection Video"
                ),

                data=processed_video,

                file_name=(
                    "tennis_detection_result.mp4"
                ),

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
            # =================================================
            # CREATE RAW OUTPUT VIDEO
            # =================================================

            raw_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )


            raw_output.close()


            raw_output_path = (
                raw_output.name
            )


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

                os.remove(
                    input_path
                )

                os.remove(
                    raw_output_path
                )

                st.stop()


            # =================================================
            # PROGRESS BAR
            # =================================================

            progress_bar = st.progress(
                0
            )


            status_text = st.empty()


            frame_count = 0


            # =================================================
            # PROCESS VIDEO FRAME BY FRAME
            # =================================================

            with st.spinner(
                "Running YOLO detection..."
            ):

                while True:

                    # -----------------------------------------
                    # Read frame
                    # -----------------------------------------

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

                    annotated_frame = (
                        result.plot()
                    )


                    # -----------------------------------------
                    # Keep original dimensions
                    # -----------------------------------------

                    if (
                        annotated_frame.shape[1]
                        != width
                        or
                        annotated_frame.shape[0]
                        != height
                    ):

                        annotated_frame = cv2.resize(
                            annotated_frame,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )


                    # -----------------------------------------
                    # Write processed frame
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
                            frame_count
                            / total_frames
                        )


                        progress_bar.progress(
                            min(
                                progress,
                                1.0
                            )
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


            progress_bar.progress(
                1.0
            )


            status_text.success(
                "YOLO detection completed!"
            )


            # =================================================
            # CONVERT TO WEB-FRIENDLY FORMAT
            # =================================================

            st.info(
                "Preparing detection video "
                "for web playback..."
            )


            final_output = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )


            final_output.close()


            final_output_path = (
                final_output.name
            )


            # =================================================
            # GET FFMPEG
            # =================================================

            ffmpeg_path = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )


            # =================================================
            # FFMPEG COMMAND
            # =================================================

            command = [

                ffmpeg_path,

                "-y",

                "-i",
                raw_output_path,

                # --------------------------------------------
                # Keep original dimensions
                # --------------------------------------------

                "-vf",
                (
                    "scale="
                    "trunc(iw/2)*2:"
                    "trunc(ih/2)*2"
                ),

                # --------------------------------------------
                # H.264
                # --------------------------------------------

                "-c:v",
                "libx264",

                # --------------------------------------------
                # Browser compatible pixel format
                # --------------------------------------------

                "-pix_fmt",
                "yuv420p",

                # --------------------------------------------
                # Optimize MP4 for streaming
                # --------------------------------------------

                "-movflags",
                "+faststart",

                # --------------------------------------------
                # Keep FPS
                # --------------------------------------------

                "-r",
                str(fps),

                final_output_path
            ]


            # =================================================
            # RUN FFMPEG
            # =================================================

            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )


            # =================================================
            # CHECK FFMPEG
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


                # --------------------------------------------
                # Cleanup
                # --------------------------------------------

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


                st.stop()


            # =================================================
            # READ FINAL VIDEO
            # =================================================

            with open(
                final_output_path,
                "rb"
            ) as f:

                processed_video = (
                    f.read()
                )


            # =================================================
            # DETECTION RESULT
            # =================================================

            st.subheader(
                "🎾 Detection Result"
            )


            # =================================================
            # ORIGINAL + DETECTION SIDE BY SIDE
            # =================================================

            col1, col2 = st.columns(
                2
            )


            # -------------------------------------------------
            # ORIGINAL VIDEO
            # -------------------------------------------------

            with col1:

                st.markdown(
                    "### 🎥 Original Video"
                )


                st.video(
                    original_video_bytes
                )


            # -------------------------------------------------
            # DETECTION VIDEO
            # -------------------------------------------------

            with col2:

                st.markdown(
                    "### 🎾 Detection Video"
                )


                st.video(
                    processed_video
                )


            # =================================================
            # DOWNLOAD BUTTON
            # =================================================

            st.download_button(
                label=(
                    "⬇️ Download Detection Video"
                ),

                data=processed_video,

                file_name=(
                    "tennis_detection_result.mp4"
                ),

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

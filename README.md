# Tennis Object Detection

**YOLO11 · Transfer Learning · Image & Video Detection**

## Project Overview

This project is a YOLO11-based computer vision system designed to detect important objects in tennis match footage.

The final model detects four classes:

* Player
* Tennis Ball
* Net
* Court

The system supports both image and video inference and returns bounding boxes, class labels, and confidence scores.

A Streamlit web application is also included for interactive image and video detection.

## Dataset

### Source

The dataset was obtained from **TennisBallTracker** by Denica Tran through Roboflow Universe.

* **Workspace:** denica-tran
* **Project:** tennisballtracker
* **Version:** 9
* **Export format:** YOLOv8
* **License:** CC BY 4.0
* **Source:** [TennisBallTracker - v9](https://universe.roboflow.com/denica-tran/tennisballtracker/dataset/9)

### Classes

The original dataset contained 15 classes. It was filtered to four classes relevant to this project:

* player
* tennis ball
* net
* court

### Dataset Split

| Split      | Images | Bounding Boxes |
| ---------- | -----: | -------------: |
| Train      |  3,273 |          8,875 |
| Validation |    457 |          1,250 |
| Test       |    229 |            609 |

### Dataset Challenge

The dataset has a significant class imbalance.

The tennis ball represents only a small portion of the training annotations and is also the smallest and most motion-blurred object. Therefore, tennis ball detection is the main challenge of the project.

## Project Workflow

### 1. Dataset Preparation

The TennisBallTracker dataset was downloaded and filtered to the four target classes:

* player
* tennis ball
* net
* court

### 2. Model Training

YOLO11n was fine-tuned using the prepared tennis dataset with COCO-pretrained weights.

### 3. Model Evaluation

The model was evaluated using:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* Confusion Matrix

### 4. Testing

The final model was tested on unseen images from the held-out test split to examine real-world detection performance.

### 5. Deployment

The trained `best.pt` model was integrated into a Streamlit application for image and video inference.

## Model

The project uses **YOLO11n (YOLO11 Nano)** with transfer learning from COCO-pretrained weights.

| Setting                 | Value        |
| ----------------------- | ------------ |
| Base checkpoint         | `yolo11n.pt` |
| Pretrained weights      | COCO         |
| Epochs                  | 30           |
| Image Size              | 1280         |
| Batch Size              | 8            |
| Early Stopping Patience | 5            |

## Model Results

The final model was evaluated on the validation split using `imgsz=1280`.

| Class       | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ----------- | --------: | -----: | ------: | -----------: |
| All Classes |     0.893 |  0.861 |   0.880 |        0.668 |
| Player      |     0.944 |  0.947 |   0.974 |        0.751 |
| Tennis Ball |     0.783 |  0.522 |   0.575 |        0.410 |
| Net         |     0.955 |  0.992 |   0.988 |        0.844 |
| Court       |     0.892 |  0.984 |   0.985 |        0.666 |

The model achieved an overall **mAP@0.5 of 0.880**.

The strongest results were obtained for the net, court, and player classes. Tennis ball detection remained the main weakness, with a recall of **0.522** and mAP@0.5 of **0.575**.

## Evaluation Visualizations

The project includes:

* `confusion_matrix.png`
* `results.png`

## Testing on Unseen Images

The final model was also tested on the held-out test split.

### Success Case

A close-up frame where the tennis ball was relatively large was correctly detected with a confidence of 0.77.

### Failure Case

A wide broadcast shot showed limitations in detecting small objects:

* The tennis ball was small and close to the net.
* A distant player was missed.

This is consistent with the model's lower performance on small and difficult objects, particularly the tennis ball.

## Streamlit Web Application

The project includes a Streamlit application for interactive image and video inference.

### Image Detection

Users can:

* Upload a tennis image.
* Adjust the confidence threshold.
* Select the inference image size.
* Run object detection.
* View the annotated image.
* View detected classes and confidence scores.

### Video Detection

Users can:

* Upload a tennis video.
* View the original video.
* Run YOLO11 detection frame by frame.
* Generate an annotated detection video.
* View the processed video in the application.
* Download the processed video.

The output video preserves the original aspect ratio and is converted to a browser-compatible MP4 format.

## Inference Settings

### Confidence Threshold

Default: `0.25`

Controls the minimum confidence required for a detection to be displayed.

### Inference Image Size

Available options:

* `640`
* `960`
* `1280`

The final model was trained using `imgsz=1280`.

## Detection Pipeline

### Image Pipeline

```text
Input Image
     ↓
YOLO11 Model
     ↓
Object Detection
     ↓
Bounding Boxes
     ↓
Class Labels
     ↓
Confidence Scores
     ↓
Annotated Image
```

### Video Pipeline

```text
Input Video
     ↓
Extract Video Frames
     ↓
YOLO11 Detection
     ↓
Draw Bounding Boxes
     ↓
Reconstruct Video
     ↓
MP4 Conversion
     ↓
Display / Download Result
```
## Requirements

```text
streamlit
ultralytics
opencv-python-headless
pillow
imageio-ffmpeg
numpy<2
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Enter the Project Directory

```bash
cd tennisvision-yolo
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser.

## Project Structure

```text
tennisvision-yolo/
│
├── README.md
├── app.py
├── best.pt
├── requirements.txt
│
├── confusion_matrix.png
├── results.png
│
└── prediction_images/
```

## Deployment

The trained model is integrated into a Streamlit web application.

```text
GitHub Repository
        ↓
Streamlit Cloud
        ↓
app.py
        ↓
best.pt
        ↓
requirements.txt
        ↓
Tennis Object Detection Web App
```

**Live Demo:** [Tennis Object Detection · Streamlit](https://tennisvision-yolo-6fbb4tct7tzf3wddcjkj6g.streamlit.app/)

## Future Improvements

* Improve tennis ball detection for very small objects.
* Increase the number of tennis ball training examples.
* Estimate tennis ball trajectory.
* Analyze player movement.
* Detect court lines.
* Develop advanced tennis performance analytics.

## Author

Sarah Tawfiq

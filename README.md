# 🎾 Tennis Ball & Player Detection

Object Detection Final Project — Computer Vision Track
**DOTPY Academy — DSA 2026**

**YOLO11 · Transfer Learning · Image & Video Detection**

---

## 📌 Project Overview

This project is a YOLO11-based computer vision system designed to detect important objects in tennis match footage.

The final model detects four classes in a single frame:

- 👤 Player
- 🎾 Tennis Ball
- 🥅 Net
- 🎾 Court

The system can process both images and videos and return bounding boxes, class labels, and confidence scores for detected objects.

The project also includes a Streamlit web application for interactive image and video inference.

---

## 🎯 What It Does

Given an image or video, the model detects:

- Tennis players
- Tennis balls
- Tennis nets
- Tennis court regions

For each detected object, the model returns:

- Bounding box
- Class label
- Confidence score

Example:

```text
tennis ball   0.77   [x1, y1, x2, y2]
player        0.91   [x1, y1, x2, y2]
```

## 🗂️ Dataset

### Source

The dataset was obtained from **TennisBallTracker** by Denica Tran through Roboflow Universe.

- **Workspace:** denica-tran
- **Project:** tennisballtracker
- **Version:** 9
- **Export format:** YOLOv8
- **License:** CC BY 4.0
- **Link:** [TennisBallTracker - v9 2023-10-05 1:16am](https://universe.roboflow.com/denica-tran/tennisballtracker/dataset/9)

Attribution is required under the dataset license.

### Classes Used

The original dataset contained 15 classes.

For this project, the dataset was filtered to the four classes relevant to tennis object detection:

- player
- tennis ball
- net
- court

### Dataset Split

| Split | Images | Bounding Boxes |
|---|---|---|
| Train | 3,273 | 8,875 |
| Validation | 457 | 1,250 |
| Test | 229 | 609 |

### Dataset Challenge

The dataset has a significant class imbalance.

The tennis ball has approximately 5–6 times fewer training instances than the other classes and is also the physically smallest and most motion-blur-prone object.

This makes tennis ball detection the main challenge of the project.

## 🧠 Model

The project uses YOLO11n (YOLO11 Nano) with transfer learning from COCO-pretrained weights.

Two training runs were performed.

| Setting | Run 1 — Baseline | Run 2 — Final |
|---|---|---|
| Base checkpoint | yolo11n.pt | yolo11n.pt |
| Pretrained | COCO | COCO |
| Epochs | 30 | 30 |
| Image Size | 960 | 1280 |
| Batch Size | 16 | 8 |
| Early-stop Patience | 8 | 5 |

### Why 1280?

The tennis ball is the smallest object in the dataset.

Increasing the image size from 960 to 1280 preserves more pixels for small objects such as the tennis ball.

The trade-off is:

- Higher computational cost
- Smaller batch size
- Slower inference

The change was made deliberately to improve tennis ball detection.

## 📊 Final Model Results

The final model was evaluated on the validation split containing 457 images using imgsz=1280.

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| All Classes | 0.893 | 0.861 | 0.880 | 0.668 |
| Player | 0.944 | 0.947 | 0.974 | 0.751 |
| Tennis Ball | 0.783 | 0.522 | 0.575 | 0.410 |
| Net | 0.955 | 0.992 | 0.988 | 0.844 |
| Court | 0.892 | 0.984 | 0.985 | 0.666 |

## 📈 Improvement from Run 1 to Run 2

Increasing the image size from 960 → 1280 improved the weakest class, tennis ball.

**Tennis Ball**

| Metric | Run 1 | Run 2 |
|---|---|---|
| Recall | 0.474 | 0.522 |
| mAP@0.5 | 0.540 | 0.575 |

This shows a measurable improvement in the detection of the smallest object in the dataset.

The tennis ball remains the most challenging class, with a recall of 0.522.

## 📊 Evaluation Visualizations

- Confusion Matrix — `confusion_matrix.png`
- Training Results — `results.png`

## 🧪 Testing on Unseen Images

The final model was also tested on the held-out test split, which was not used during training or validation.

**Success Case**

A close-up frame where the tennis ball was relatively large in the image was correctly detected with a confidence of 0.77.

**Failure Case**

A wide broadcast shot was identified as a failure case where:

- The tennis ball was small and near the net.
- A distant second player was also missed.

This visually confirmed failure is consistent with the measured weakness of the model on small objects.

## 💻 Streamlit Web Application

The project includes a Streamlit application that allows users to interact with the trained YOLO11 model.

### 🖼️ Image Detection

Users can:

- Upload a tennis image.
- Adjust the confidence threshold.
- Select the inference image size.
- Run object detection.
- View the annotated image.
- View detected classes and confidence scores.

### 🎥 Video Detection

The application also supports video detection.

Users can:

- Upload a tennis video.
- View the original uploaded video.
- Run YOLO11 detection frame by frame.
- Generate an annotated detection video.
- View the detection video directly inside the website.
- Download the processed detection video.

The detection video maintains the original aspect ratio and is converted to a browser-compatible MP4 format.

## ⚙️ Inference Settings

The application provides two configurable inference settings.

**Confidence Threshold**

Default: `0.25`

The confidence threshold controls the minimum confidence required for a detection to be displayed.

**Inference Image Size**

Available options: `640`, `960`, `1280`

The final model was trained using `imgsz = 1280`.

## 🔬 Detection Pipeline

**Image Pipeline**

```
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

**Video Pipeline**

```
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
H.264 Conversion
     ↓
Browser-Compatible MP4
     ↓
Display Detection Video
     ↓
Download Result
```

## 🛠️ Technologies Used

- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- Pillow
- FFmpeg
- NumPy
- Google Colab

## 📦 Requirements

The deployment uses:

```
streamlit
ultralytics
opencv-python-headless
pillow
imageio-ffmpeg
numpy<2
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 🚀 How to Run

**1. Clone the Repository**

```bash
git clone <your-repository-url>
```

**2. Enter the Project Directory**

```bash
cd tennisvision-yolo
```

**3. Install Requirements**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit App**

```bash
streamlit run app.py
```

The application will then open in your browser.

## 📁 Project Structure

```
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
├── pred_1360f85d7fabbe56_jpg.rf....jpg
├── pred_maxresdefault-2-_jpg.rf....jpg
├── pred_videoplayback_mp4-0_jpg.rf....jpg
├── pred_youtube-10_jpg.rf....jpg
└── pred_youtube-13_jpg.rf....jpg
```

## 📈 Model Performance Summary

The final model achieved:

**Overall**
```
Precision       = 0.893
Recall          = 0.861
mAP@0.5         = 0.880
mAP@0.5:0.95    = 0.668
```

**Best Performing Classes**

The model performs particularly well on:

- Player
- Net
- Court

The strongest mAP@0.5 results are:

```
Net       = 0.988
Court     = 0.985
Player    = 0.974
```

**Main Challenge**

The tennis ball remains the most difficult class:

```
Precision       = 0.783
Recall          = 0.522
mAP@0.5         = 0.575
mAP@0.5:0.95    = 0.410
```

This is mainly related to its small size, motion blur, and class imbalance in the dataset.

## 🌐 Deployment

The final model is integrated into a Streamlit web application.

The deployment uses:

```
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
Tennis Detection Web App
```

**Live Demo**

🔗 Streamlit App: [Tennis Object Detection · Streamlit](https://tennisvision-yolo-6fbb4tct7tzf3wddcjkj6g.streamlit.app/)

## 🎯 Application Features

| Feature | Status |
|---|---|
| Image Upload | ✅ |
| Image Detection | ✅ |
| Video Upload | ✅ |
| Video Detection | ✅ |
| Player Detection | ✅ |
| Tennis Ball Detection | ✅ |
| Net Detection | ✅ |
| Court Detection | ✅ |
| Confidence Threshold | ✅ |
| Inference Image Size | ✅ |
| Detection Video Preview | ✅ |
| Download Detection Video | ✅ |
| Browser-Compatible MP4 | ✅ |

## 🔮 Future Improvements

Possible future improvements include:

- Improving tennis ball detection for very small objects.
- Increasing the number of tennis ball training examples.
- Object tracking across video frames.
- Tennis ball trajectory estimation.
- Player movement analysis.
- Court line detection.
- Hit detection.
- Serve analysis.
- Real-time webcam inference.
- Advanced tennis performance analytics.

## 📚 Project Workflow

**1. Dataset Preparation**

The TennisBallTracker dataset was downloaded and filtered to the four target classes:

- player
- tennis ball
- net
- court

**2. Model Training**

YOLO11n was fine-tuned using the prepared tennis dataset.

**3. Model Evaluation**

The model was evaluated using:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Confusion Matrix

**4. Model Improvement**

The input resolution was increased from 960 → 1280 to improve small-object detection.

**5. Unseen Testing**

The final model was tested on the held-out test split.

**6. Deployment**

The trained `best.pt` model was integrated into a Streamlit application for image and video inference.

## 👩‍💻 Author

*[Your name here]*

Computer Science & Artificial Intelligence
DOTPY Academy — DSA 2026

## ⭐ Project

Tennis Ball & Player Detection

Computer Vision Track — YOLO11 Object Detection

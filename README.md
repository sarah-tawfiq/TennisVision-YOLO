# 🎾 Tennis Object Detection

YOLO11-based object detection system for tennis videos and images.

The model detects four classes:

- Player
- Tennis Ball
- Net
- Court

## 🚀 Features

- Image object detection
- Video object detection
- Bounding boxes and confidence scores
- Downloadable processed videos
- Streamlit web interface

## 🧠 Model

YOLO11n was trained for tennis object detection.

### Classes

| Class | mAP50 |
|------|------:|
| Player | 0.975 |
| Tennis Ball | 0.540 |
| Net | 0.987 |
| Court | 0.990 |

Overall validation results:

- Precision: 0.887
- Recall: 0.842
- mAP50: 0.873
- mAP50-95: 0.696

## 🛠️ Technologies

- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- PyTorch

## ▶️ Run Locally

Install dependencies:

```bash
pip install -r requirements.txt

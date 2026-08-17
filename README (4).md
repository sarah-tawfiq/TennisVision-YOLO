# Tennis Ball & Player Detection

Object Detection Final Project — Computer Vision Track (DOTPY Academy)
YOLO11 · Transfer Learning · Real-Time Detection

A YOLO11-nano detector fine-tuned on tennis match footage to locate **players**, the **tennis ball**, the **net**, and the **court** in a single frame — turning a video a coach would otherwise rewatch by eye into labelled coordinates.

## What it does

Given an image (or a video frame), the model returns a bounding box, class label, and confidence score for every player, tennis ball, net, and court region it finds.

```
tennis ball  0.77   [x1, y1, x2, y2]
player       0.91   [x1, y1, x2, y2]
```

## Dataset

- **Source:** [TennisBallTracker](https://universe.roboflow.com/denica-tran/tennisballtracker) by Denica Tran — Roboflow Universe, workspace `denica-tran`, project version 9 (YOLOv8 export). **Licence: CC BY 4.0** (attribution required).
- **Classes used:** `player`, `tennis ball`, `net`, `court` (filtered down from the original 15-class dataset, which also included body-part/keypoint classes not relevant to this task).
- **Split:** Train 3,273 images / 8,875 boxes · Validation 457 images / 1,250 boxes · Test 229 images / 609 boxes.
- **Known weakness:** heavy class imbalance — `tennis ball` has ~5–6x fewer training instances than any other class (549 vs. 2,626–3,012), and is also the physically smallest, most motion-blur-prone object in the set.
- **Minor data-quality note:** a few test-split images contain visible pixel noise / colour distortion, most likely from augmentation applied during the Roboflow export rather than anything introduced during this project. Flagged rather than hidden — see `SRS_Documentation.pdf` Section 13.

## Model

Two training runs were done, the second a deliberate iteration on the first:

| Setting | Run 1 (baseline) | Run 2 (final, used for submission) |
|---|---|---|
| Base checkpoint | `yolo11n.pt` (COCO-pretrained) | `yolo11n.pt` (COCO-pretrained) |
| Epochs | 30 | 30 (completed in full) |
| Image size | 960 | **1280** |
| Batch size | 16 | 8 (reduced to fit GPU memory at 1280px) |
| Early-stop patience | 8 | 5 |

**Why the change:** the tennis ball is the smallest object in the dataset by a wide margin. Raising `imgsz` to 1280 preserves more pixels-on-target for it, at the cost of a smaller batch size and slower training/inference — a trade made on purpose because the ball is the class this project cares about most.

## Results (validation split, 457 images) — final model, imgsz=1280

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| All classes | 0.893 | 0.861 | 0.880 | 0.668 |
| player | 0.944 | 0.947 | 0.974 | 0.751 |
| **tennis ball** | 0.783 | **0.522** | **0.575** | 0.410 |
| net | 0.955 | 0.992 | 0.988 | 0.844 |
| court | 0.892 | 0.984 | 0.985 | 0.666 |

**Iteration evidence:** the resolution change from 960→1280 measurably improved the weakest class — tennis ball recall rose from 0.474 to 0.522, and mAP@0.5 from 0.540 to 0.575. It is still the lowest-scoring class (the model still misses roughly 1 in 2 real balls), but the improvement is real and measured, not theoretical. Full interpretation, confusion matrix, and failure-case analysis are in `SRS_Documentation.pdf`.

Inference speed increases at 1280px vs. 960px, as expected — this must be re-measured on whatever machine the live demo runs on before claiming real-time performance there.

## Testing on unseen images

Predictions were run on the held-out test split (never used in training or validation):

- **Success case:** a close-up frame where the ball is large in the image — detected correctly at 0.77 confidence.
- **Failure case:** a wide broadcast shot where the ball (small, near the net) and a distant second player were both missed entirely — a direct, visually confirmed example of the small-object weakness measured above, not a guess. See `results/unseen_predictions/` and the SRS for the annotated images.

## How to run

**1. Train / reproduce (Colab):**
Open `notebook.ipynb` in Google Colab, set a `ROBOFLOW_API_KEY` in Colab secrets, and run all cells top to bottom. This downloads the dataset, filters it to the 4 target classes, trains YOLO11n at imgsz=1280, evaluates on the validation split, and runs inference on the held-out test images.

**2. Run the demo app:**
```bash
pip install -r requirements.txt
streamlit run app.py
```
*(Replace `app.py` with the actual demo script filename.)*

## Folder structure

```
FirstName_LastName_ObjectDetection/
├── README.md
├── SRS_Documentation.pdf
├── notebook.ipynb
├── requirements.txt
├── model/
│   └── best.pt
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── unseen_predictions/
└── demo/
    └── demo_screenshots or demo_video.mp4
```

## Still to add before submission

- [x] Exact Roboflow project URL + licence in the SRS Dataset section
- [x] `best.pt` trained weights file in `model/`
- [x] Annotated unseen test images (with confidence scores visible) in `results/unseen_predictions/`, including a confirmed failure case
- [ ] Streamlit demo app — confirm it runs reliably and add a screenshot or short recording to `demo/`


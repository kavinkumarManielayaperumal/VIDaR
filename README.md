# **Real-Time Object Detection with YOLOv5 & YOLOv8**

## **Project Overview**

This project presents a modular and extensible pipeline for real-time object detection using **YOLOv5** and **YOLOv8**, with support for camera-based input and custom datasets. Leveraging pretrained YOLO models (trained on the **COCO dataset**), the system processes incoming images from either static datasets or live camera feeds, feeding them through a customized preprocessing pipeline into the YOLO model.

## **Motivation**

Initially, the goal was to build an object detection model from scratch. However, the complexity and training demands of such deep neural networks make this approach highly time-consuming. Instead, this project uses **YOLO (You Only Look Once)**, a state-of-the-art, real-time object detection architecture. By developing a **custom data loader**, the project bridges raw image input (from files or camera) and model inference with minimal setup.

## **Key Features**

- 🔧 **Custom Data Loader**: Automatically preprocesses images (resizing, normalizing, annotation handling), and formats them for YOLO input.
- 📷 **Camera Integration**: Supports live image capture and real-time detection using webcam or external camera devices.
- 🧠 **YOLOv5 & YOLOv8 Support**: Flexible architecture supporting both versions with pretrained weights on COCO.
- 📁 **Dataset Flexibility**: Works with COCO-style datasets or custom annotated images in YOLO format.

---

## **Pipeline Components**

### 1. **Data Loading & Annotation Handling**
- Loads images from dataset folders or captures from a live camera.
- If using dataset images:
  - Supports COCO-format annotations and converts them to YOLO format.
  - If annotations are missing, allows for manual labeling.
- Converts bounding boxes into YOLO-style normalized format:  
  `[class_id, x_center, y_center, width, height]`

### 2. **Image Preprocessing**
- Resizes all input images to a fixed resolution (e.g., `640x640`) as required by the YOLO architecture.
- Applies PyTorch transforms:
  - Tensor conversion
  - Normalization
  - Optional data augmentation

### 3. **Model Inference**
- Uses pretrained **YOLOv5** or **YOLOv8** models (trained on COCO dataset).
- Accepts tensor images from the custom data loader.
- Outputs bounding boxes, class predictions, and confidence scores.

---

## **Technologies Used**

- **Python**
- **PyTorch**
- **Ultralytics YOLOv5 / YOLOv8**
- **OpenCV** (for camera integration)
- **pycocotools** (for COCO dataset parsing)
- **Matplotlib / Seaborn** (for visualization)

---

## **How to Run**

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VIDaR.git
cd VIDaR
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Detection on Camera Input
```bash
python detect_camera.py --model yolov8n.pt
```

### 4. Run Detection on Custom Dataset
```bash
python train.py --img 640 --batch 16 --epochs 50 --data path/to/data.yaml --weights yolov5s.pt
```

---

## **Folder Structure**

```bash
/project-root
│
├── data_loader/
│   └── custom_loader.py       # Custom dataset + camera loader
│
├── models/
│   └── yolov5/                # YOLOv5 files
│   └── yolov8/                # YOLOv8 integration
│
├── datasets/
│   ├── images/
│   ├── labels/
│
├── detect_camera.py          # Live camera detection script
├── train.py                  # Model training script
├── README.md
└── requirements.txt
```

---

## **Next Steps**

- Expand support for instance segmentation
- Integrate edge-device compatibility (e.g., Jetson Nano, Raspberry Pi)
- Add GUI interface for live annotation and feedback

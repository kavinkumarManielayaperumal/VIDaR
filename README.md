# **Object Detection with YOLO on the COCO Dataset**

## **Project Overview**

This project focuses on implementing **object detection** using the **YOLO (You Only Look Once)** algorithm on the **COCO dataset**. The goal of the project was to understand how data from the COCO dataset can be preprocessed, formatted, and loaded into a **YOLO model** for object detection.

The project showcases how to:
- **Load and preprocess the COCO dataset**.
- **Extract and normalize bounding boxes** from COCO annotations.
- **Convert the dataset into the YOLO format** for object detection.
- **Train a YOLO model** (theoretically, if required) on the custom data.

## **Key Concepts & Steps**:
1. **Dataset Loading**:
   - The COCO dataset consists of images with annotations that define objects in the form of **bounding boxes**.
   - We manually loaded the **COCO annotations** and reshaped them into **YOLO format** (class ID, normalized bounding box coordinates).
   
2. **Data Preprocessing**:
   - **Resized images** to a common size (224x224).
   - **Normalized bounding boxes**: The bounding boxes were rescaled to be **relative** to the image size (in YOLO format, `[x_center, y_center, width, height]`).
   - **PyTorch transforms** were applied to convert images to tensors, normalize them, and apply any necessary augmentations.

3. **YOLO Data Conversion**:
   - The **COCO annotations** were converted into **YOLO format** for each image. This includes converting bounding box coordinates into **center-based format** and **rescaling them**.

4. **YOLOv5 Integration**:
   - We used **YOLOv5** for object detection, which was **pre-trained** on the COCO dataset. The project provides a step-by-step guide for **fine-tuning YOLOv5** with the COCO dataset annotations.
   - Instructions for **training** the model and **evaluating** its performance are included.

## **Technologies Used**:
- **Python**
- **PyTorch**
- **torchvision** (for models like YOLO)
- **COCO Dataset**
- **YOLOv5**
- **pycocotools** (for COCO dataset handling)

## **How the YOLO Model Was Trained**:
1. **Preprocessing**: COCO annotations were converted into the YOLO format (class ID, bounding boxes with center coordinates).
2. **Dataset Loading**: The processed images and annotations were loaded into **PyTorch DataLoader**.
3. **Model Fine-Tuning**: YOLOv5 was fine-tuned on the custom dataset for object detection.
4. **Evaluation**: Model performance was evaluated using standard **precision**, **recall**, and **mAP** metrics on a validation set.

## **How to Run**:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/YOLOv5-COCO-ObjectDetection.git
   cd YOLOv5-COCO-ObjectDetection
2. install dependencies:
   ```bash
   pip install -r requirements.txt
3. Prepare the dataset (refer to the Data Preprocessing section).
4. Train the model (YOLOv5):
    ```bash
	python train.py --img 640 --batch 16 --epochs 50 --data path/to/data.yaml --weights yolov5s.pt --cache
5. Evaluate the model:
   ```bash
   python val.py --weights path/to/best_model.pt --data path/to/data.yaml --img 640

## **Folder Structure**:
   ```bash
   /dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            val_image1.jpg
            val_image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            val_image1.txt
            val_image2.txt
            ...
    /yolov5
       /models
       /data
       /runs
       /... (YOLOv5 files).






 
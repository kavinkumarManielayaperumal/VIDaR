from ultralytics import YOLO
import torch

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# move the model to gpu
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="coco8.yaml",
    epochs=600,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=device,  
)

metrics = model.val()

# save the trained model , simply saving the wights of the model , for the test on custom dataset
model.save("yolo11n_trained.pt")


# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model
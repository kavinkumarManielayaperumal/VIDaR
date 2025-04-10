# now we will use the yolo model to predict the bounding box and the label of the image 


import torch
import torch.nn as nn 
from ultralytics import YOLO
import torchvision.transforms as T
from data_transformer import getdataloader

device= torch.device('cuda'if torch.cuda.is_available()else 'cpu')

# here we are using the pretrained model of the yolov8n model which is trained on the coco dataset 
 # but we will train the model on our own dataset but it more efficicent to use the pretrained model 
 
 
 # usually do this , model.laod_state_dict(torch.load("____.pt ")) like this we will load the torch file , 
 # but in this case we are using the ultralytics library which is a wrapper around the pytorch model
def yolo_model(getdataloader,device):
   model=YOLO("yolov8n.pt") # here we are using the yolov8n model which is the smalles model of the yolov8
   for image_tensor,label_tensor,box_tensor , image_id_tensor in enumerate(getdataloader):
       image_tensor=image_tensor.to_device(device)
       label_tensor=label_tensor.to_device(device)
       box_tensor=box_tensor.to_device(device)
       image_id_tensor=image_id_tensor.to_device(device)
       # now we will use the model to predict the bounding box and the label of the image
       pred=model(image_tensor)
       
       # here we will get the bounding box and the label of the image
       box=pred.xyxy[0]
       for i in box:
           x1,y1,x2,y2,conf,cls=i
           print(f"Bounding box :{x1},{y1},{x2},{y2},confidence:{conf},class:{cls}")
       pred.show()
       pred.save("predictions")
       
       
       
       
if __name__=="__main__":
    annotations_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    images_path=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017" \
    
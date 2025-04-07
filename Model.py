# now we will use the yolo model to predict the bounding box and the label of the image 


import torch
import torch.nn as nn 
from ultralytics import YOLO
import torchvision.transforms as T

model=YOLO("yolov8n.pt") # here we are using the yolov8n model which is the smalles model of the yolov8

# here we are using the pretrained model of the yolov8n model which is trained on the coco dataset 
 # but we will train the model on our own dataset but it more efficicent to use the pretrained model 
 
 
 # usually do this , model.laod_state_dict(torch.load("____.pt ")) like this we will load the torch file 
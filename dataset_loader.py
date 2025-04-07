
# this code is used to load the dataset from the coco dataset but  the cathses the image will loading will be problematic beacuse the 30000 image loaded and stored in the memory so dont use this code 
#this not relevant to our project
import os
import json
from pycocotools.coco import COCO
from PIL import Image
from torchvision import transforms as T
import torch 
def extract_image_and_bounding_box(annotation_file, image_dir):
    coco=COCO(annotation_file)
    image_ids=list(coco.imgs.keys())
    print(f"Total number of images:{len(image_ids)}")
    
    for image_id in image_ids:
        image_info=coco.loadImgs(image_id)[0]
        file_name=image_info['file_name']
        image_path=os.path.join(image_dir, file_name)
        
        # load the image in PIL format   
        input_image=Image.open(image_path)
        input_image=input_image.convert("RGB")# convert the image to RGB format
        
        
        
        # now we will get the annotations of the image 
        annotation_file_id=coco.getAnnIds(imgIds=image_id)
        annotation_file_load=coco.loadAnns(annotation_file_id)
        
        # now we will get the bounding box of the each annotation of the image 
        box=[]# we need to the bounding box of the image so that we can use it in the model
        label=[]
        for ann in annotation_file_load:
            bbox=ann['bbox']
            category_id =ann['category_id']
            
            x,y,width,hight=bbox
            box.append([x,y,width,hight])
            label.append(category_id)
            
      
    return input_image,box,label,image_id




if __name__ == "__main__":
    annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    image_dir=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017"
    
    input_image,box_tensor,label_tensor,image_id=extract_image_and_bounding_box(annotation_file, image_dir)
    for i in range(len(input_image)):
        print(f"Image ID: {image_id[i]}, Image Tensor: {input_image[i]}, Bounding Box: {box_tensor[i]}, Label: {label_tensor[i]}")
        if i==5:
            break
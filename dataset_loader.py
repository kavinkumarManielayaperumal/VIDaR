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
        image=Image.open(image_path)
        transform=T.ToTensor()# this will convert the image to tensor format (c,h,w), exactly cifar10 dataset format 
        image_tensor=transform(image)# this will convert the image to tensor format (c,h,w) , where c is the channel, h is the height and w is the width of the image
        
        # this is important because this is the input for the model , like the model will take the image in this format
        
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
            
        # now we will convert the bounding box to tensor format
        box_tensor=torch.tensor(box,dtype=torch.float32)
        label_tensor=torch.tensor(label,dtype=torch.int64)
        image_id=torch.tensor(image_id,dtype=torch.int64)
    return image_tensor,box_tensor,label_tensor,image_id




if __name__ == "__main__":
    annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    image_dir=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017"
    
    image_tensor,box_tensor,label_tensor,image_id=extract_image_and_bounding_box(annotation_file, image_dir)
    for i in range(len(image_tensor)):
        print(f"Image ID: {image_id[i]}, Image Tensor: {image_tensor[i]}, Bounding Box: {box_tensor[i]}, Label: {label_tensor[i]}")
        if i==5:
            break
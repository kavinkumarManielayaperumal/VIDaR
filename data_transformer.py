import torch
import torch.functional as F
from torch.utils.data import Dataset,DataLoader
import os

from torchvision import transforms as T
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# first, we need to rehape the image into comman size, then after that we will load the data into the dataloader

class ResizeAndNormalization:
    def __init__(self,image:Image.Image,box,image_size=(224,224)):
        # here we are using the image:Image.Image which is the PIL image format
        self.image_size= image_size
        self.image,self.box=image,box
        
    def __call__(self):
        image_resized=self.image.resize(self.image_size)
        
        #now we will rehape the boundinf box into the same size as the image 
        original_width,original_height=self.image.size
        new_width,new_height=self.image_size
        
        #scale treh bounding box to the new size
        scale_x= new_width/original_width
        scale_y= new_height/original_height
        
        new_box=[]
        
        for i in range(len(self.box)):
            x,y,width, height=self.box[i]
            new_x=int(x*scale_x)
            new_y=int(y*scale_y)
            new_widths=int(width*scale_x)
            new_heights=int(height*scale_y)
            
            new_box.append([new_x,new_y,new_widths, new_heights])
          
        return image_resized,new_box
       

class CocoDatasetTransform(Dataset):
    def __init__(self, annotation_file,image_dir, transform=None):
        self.coco= COCO(annotation_file)
        self.image_id=list(self.coco.imgs.keys())
        self.image_dir=image_dir
        self.transform=transform if transform else T.ToTensor()
        
    def __len__(self):
        return len(self.image_id)
    def __getitem__(self ,index):
        image_id=self.image_id[index]
        image_info=self.coco.loadImgs(image_id)[0]
        image_file_name=image_info['file_name']
        image_path=os.path.join(self.image_dir,image_file_name)
        image_id=image_info["id"]
        
        # load the image in PIL format
        input_image=Image.open(image_path)
        input_image=input_image.convert("RGB")
        
        #now we will get the annotations of the image
        annotation_file_id=self.coco.getAnnIds(imgIds=image_id)
        annotation_file_load=self.coco.loadAnns(annotation_file_id)
    
        # now we will get the bounding box of the each annotation of the image 
        
        box=[]
        label=[]
        for ann in annotation_file_load:
            bbox=ann['bbox']
            category_id=ann['category_id']
            x,y,width,height=bbox
            box.append([x,y,width,height])
            label.append(category_id)
         
        resized=ResizeAndNormalization(input_image,box,image_size=(224,224))
        resized_image,resized_box=resized()
        
        # now we need to covert the box and lablel into single tensor with center_x,center_y,width, height formar
        # we will check the shape of the image tensor and the bow tensor and the label tensor and the image id tensor
        #print(f"Image Tensor Shape:{image_tensor.shape}, Box Tensor shape:{box_tensor.shape},Label Tensor shape:{label_tensor.shape}, Image ID Tensor shape:{image_id_tensor.shape}")
        img_w,img_h=resized_image.size
        target=[]
        for i in range(len(resized_box)):
            x,y,width,height=resized_box[i]
            cen_x=(x+width/2)/img_w
            cen_y=(y+height/2)/img_h
            width/=img_w
            height/=img_h
            target.append([label[i],cen_x,cen_y,width,height])
            
         
        
        
        # now eveything is done we will convert the image into tensor and normalize it 
        image_tensor=self.transform(resized_image)
        # here the we can use the normal tensor coversion but its in the form of (C,H,W) , so we need to use the torchvision library tensor , ToTensor()
        box_tensor=torch.tensor(resized_box,dtype=torch.float32)
        label_tensor=torch.tensor(label,dtype=torch.long)
        image_id_tensor=torch.tensor(image_id,dtype=torch.long)
        target_tensor=torch.tensor(target,float())
        
        # we will normalization the imagr tensor to the range of 0-1
        image_tensor= image_tensor/255.0 # here we are using the normalization ,so everything is in the range of 0-1
        
        # we will check the shape of the image tensor and the bow tensor and the label tensor and the image id tensor
        #print(f"Image Tensor Shape:{image_tensor.shape}, Box Tensor shape:{box_tensor.shape},Label Tensor shape:{label_tensor.shape}, Image ID Tensor shape:{image_id_tensor.shape}")
        
        return image_tensor,box_tensor,label_tensor,image_id_tensor,target_tensor
    
 
def custom_collate_fn(batch):
    images=[]
    boxes=[]
    labels=[]
    image_ids=[]
    target=[]
    
    for items in batch:
        image,box,label,image_id,target_tensor=items
        images.append(image)
        boxes.append(box)
        labels.append(label)
        image_ids.append(image_id)
        target.append(target_tensor)
    
    return torch.stack(images),boxes,labels,image_ids,target_tensor# here we are using the torch.stack to stack the image into a single tensor , so now it will be in the shape of (batch_size,c,h,w),
   # this simply we are stacking the image in single batch sizze 
   

    

    
# now we will use the dataloader to load the data into the model 
def getdataloader(annotation_file,image_dir,batch_size=32,shuffle=True):
    dataset_loader=CocoDatasetTransform(annotation_file,image_dir)
    dataloader=DataLoader(dataset_loader,batch_size=batch_size,shuffle=shuffle,collate_fn=custom_collate_fn)
    return dataloader
        
        
if __name__== "__main__":
    
    annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    image_dir=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017"
        
        
    image_size=(224,224)
    
    
    
    # now we will use the dataloader to load the data into the  model 
    batch_size=32
    shuffle=True
    dataloader=getdataloader(annotation_file,image_dir,batch_size=batch_size,shuffle=shuffle)
    for batch in dataloader:
       images, boxes, labels, image_ids,target_tensor = batch
       print(f"Images shape: {images.shape}")
       # box in the 
       print(f"Boxes shape: {boxes}")
       print(f"Labels shape: {labels}")
       print(f"Image IDs shape: {image_ids}")
       print(f"Target Tensor shape:{target_tensor}")
       break  # just get the first batch
    
    
    
    
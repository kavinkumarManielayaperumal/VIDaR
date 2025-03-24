
# the coco dataset have 30000 images so even one image have to region, like its have the bounding box and the segmentation of the object in the image , so the annotation of the image is very complex 
# so we will be using the pycocotools to work with the coco dataset 


from PIL import Image # this pillow library is used to work with the images 
import matplotlib.pyplot as plt 
 
from pycocotools.coco import COCO
import os 

# this is the annotation file of the coco dataset 
annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
coco=COCO(annotation_file) # this will load and map the annotation file to the coco object 



image_info= coco.loadImgs(36)[0]

print(image_info)

image_path=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017\000000000036.jpg" 
image=Image.open(image_path)

# Get the annotations of the image
image_id=image_info['id'] # this will get the id of the image of paticular the image id 36

annotations=coco.getAnnIds(imgIds=image_id) # this getAnnIDs will map the image id to the annotation id 

annotations=coco.loadAnns(annotations) # this will load the annotations of the image

fig, ax = plt.subplots(1)
ax.imshow(image)

for ann in annotations:# this will loop through the annoatation , if annoatation is more than one , then we are looking bbox key word in the annotation
    bbox=ann['bbox'] # mapping the bbox key word to the bbox
    x,y , width, height=bbox
    rect=plt.Rectangle((x,y),width,height,linewidth=1, edgecolor='r')
    ax.add_patch(rect) # add_patch is used to add the rectangle to the image 
    
plt.axis('off')
plt.show()

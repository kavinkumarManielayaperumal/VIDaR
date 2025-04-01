from pycocotools.coco import COCO


# the coco dataset have 30000 images so even one image have to region, like its have the bounding box and the segmentation of the object in the image , so the annotation of the image is very complex 
# so we will be using the pycocotools to work with the coco dataset 


from PIL import Image # this pillow library is used to work with the images 
import matplotlib.pyplot as plt





def image_viewer(annotation_file,image_path):
    coco=COCO(annotation_file) # this will load and map the annotation file to the coco object 
    image_info=coco.loadImgs(36)[0]# this will load the image of the image id36 
    print(image_info) # this will print the image info of the image id 36
    
    image=Image.open(image_path)
    fig,(ax1,ax2)=plt.subplots(1,2)# this will create the two axis for the image
    
    image_id=image_info['id'] # this will get the id of the image of paticulat the image id 36
    annotations_id=coco.getAnnIds(imgIds=image_id) # this getAnnIDs will map the image id to the annotation id
    annotation_load=coco.loadAnns(annotations_id) # this will load the annotations of the image
    for ann in annotation_load:
        bbox=ann['bbox']
        category_id=ann['category_id']
        category_name=coco.loadCats([category_id])[0]['name']# its is like this category_data = [{'id': 3, 'name': 'cat'}] so its in list format so we have to get the first element of the list and then get the name of the category
        x,y,width,height=bbox
        rect=plt.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none')
        ax2.text(x,y,f"{category_id},{category_name}",fontsize=10,color='green')
        ax2.add_patch(rect) 
    
    fig.suptitle("differece between the bounding box",fontsize=20)
    ax1.imshow(image)
    ax1.set_title("Original image")
    ax1.axis('off')
    
    ax2.imshow(image)
    ax2.set_title("Annotated image")
    ax2.axis('off')
    
    plt.show()
    
if __name__=="__main__":
    
    annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    image_path=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017\000000000036.jpg"
    
    image_viewer(annotation_file,image_path)
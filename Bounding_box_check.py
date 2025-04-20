from pycocotools.coco import COCO


# the coco dataset have 30000 images so even one image have to region, like its have the bounding box and the segmentation of the object in the image , so the annotation of the image is very complex 
# so we will be using the pycocotools to work with the coco dataset 


from PIL import Image # this pillow library is used to work with the images 
import matplotlib.pyplot as plt





def image_viewer(annotation_file,image_path):
    coco=COCO(annotation_file) # this will load and map the annotation file to the coco object 
    image_info=coco.loadImgs(109355)[0]# this will load the image of the image id36 
    print(image_info) # this will print the image info of the image id 36
    
    input_image=Image.open(image_path)
   
    
    image_id=image_info['id'] # this will get the id of the image of paticulat the image id 36
    annotations_id=coco.getAnnIds(imgIds=image_id) # this getAnnIDs will map the image id to the annotation id
    annotation_load=coco.loadAnns(annotations_id) # this will load the annotations of the image
    box=[]
    label=[]
    for ann in annotation_load:
        bbox=ann['bbox']
        category_id=ann['category_id']
        x,y,width,height=bbox
        box.append([x,y,width, height])
        label.append(category_id)
        
    print(f"Box:{box}")
    print(f"Label:{label}")
    return box,label,input_image


class resized_images():
    def __init__(self,input_image:Image.Image,box,image_size=(224,224)):
        self.input_image=input_image
        self.original_imput_image=input_image
        self.box=box
        self.image_size=image_size
    def __call__(self):
        image_resized=self.input_image.resize(self.image_size)
        
        original_width,original_height=self.input_image.size
        new_width,new_height=self.image_size
        
        scale_x=new_width/original_width
        scale_y=new_height/original_height
        new_box=[]
        for i in range(len(self.box)):
            x,y,width,height=self.box[i]
            new_x=int(x*scale_x)
            new_y=int(y*scale_y)
            new_widths=int(width*scale_x)
            new_heights=int(height*scale_y)
            new_box.append([new_x,new_y,new_widths,new_heights])
        return image_resized,new_box
    
def visualize_resized_image(annotation_file,image_path):
    coco=COCO(annotation_file)
    box,label,input_image=image_viewer(annotation_file,image_path)
    
    
    resized_image=resized_images(input_image,box,image_size=(224,224))
    image_resized,new_box=resized_image()
    #original_image=Image.open(image_path)
    #image_resized=Image.open(image_path)
    
    fig,(ax1,ax2,ax3,ax4)=plt.subplots(1,4)
    fig.suptitle("Resized image and bounding box",fontsize=20)
    for i in range(len(box)):
        x,y,width,height=box[i]
        
        rect=plt.Rectangle((x,y),width,height,linewidth=2,edgecolor="r",facecolor="None")
        category_name=coco.loadCats([label[i]])[0]["name"]
        ax2.text(x,y,f"{label[i]},{category_name}",fontsize=10,color="green")
        ax2.add_patch(rect)
        
    ax1.imshow(input_image)
    ax1.set_title("original image",fontsize=5)
    ax1.axis("off")
        
    ax2.imshow(input_image)
    ax2.set_title("original image with original bounding box ",fontsize=5)
    ax2.axis("off")

        
        
    for i in range(len(new_box)):
        x,y,width,height=new_box[i]
        rect=plt.Rectangle((x,y),width,height,linewidth=2,edgecolor="r",facecolor="None")
        category_name=coco.loadCats([label[i]])[0]["name"]
        ax4.text(x,y,f"{label[i]},{category_name}",fontsize=10,color="green")
        ax4.add_patch(rect)
        
        
        
    ax3.imshow(image_resized)
    ax3.set_title("Resized image",fontsize=5)
    ax3.axis("off")
        
    ax4.imshow(image_resized)
    ax4.set_title("Resized image with resized bounding box",fontsize=5)
    ax4.axis("off")
    plt.show()
        
        
        
    
    
            
    
    
if __name__=="__main__":
    
    annotation_file=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\annotations\instances_train2017.json"
    image_path=r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\train2017\000000109355.jpg"
    
    visualize_resized_image(annotation_file,image_path)
from PIL import Image
import matplotlib.pyplot as plt


image= Image.open(r"E:\for practice game\object detection\ObjectDetectNet\dataset\archive (1)\coco2017\test2017\000000011986.jpg")

imagecroped=image.crop((150,60,100+400,150+150))

# this orintation of the image, like (x1,y1,x2,y2) , where (x1,y1) is the top left corner and (x2,y2) is the just the height and width of the image 


# this coco dataset is stored in the format of (x,y,W,H) and it in the JSON file format , so dataset is itself have the annotation of the image

# its have the image id and cooridnates of the bounding box and the category of the object in the image, and it not only have the normal annotation but also have the segmentation of the oject in the image including so any are there
#  it will be headache to work with the coco dataset , so will be using the  pycocotools to work  with the coco dataset

plt.subplot(1,2,2)
plt.imshow(image)
plt.gca().add_patch(plt.Rectangle((100,50),100,50,linewidth=1, edgecolor='r', facecolor='none'))
plt.title("Original Image")
plt.axis("off")


plt.subplot(1,2,1)
plt.imshow(imagecroped)
plt.title("Croped image")
plt.axis("off")


plt.show()
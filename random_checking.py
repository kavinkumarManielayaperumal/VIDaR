import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
# for the image converstion we 
# Create a random RGB image (100x100 pixels, 3 channels)
np.random.seed(42)
image = np.random.rand(100, 100, 3)
image=(image*225).astype(np.uint8) # multiply by 225 gives the image range between 0-225
image=Image.fromarray(image)# convert the numpy scalar values to original PIL image format
# we already seen this in troch converstion "data=torch.from_numpy(data)" like this we will convert the numpy array to PIL image format
image_converstion=image.convert("RGB")
print(f"image shape:{np.array(image_converstion).shape}")

# Create a figure and axis
fig, (ax1,ax2) = plt.subplots(1,2) 

fig.suptitle('random image vs annotated image', fontsize=20)

ax1.imshow(image_converstion)
ax1.set_title("orginal random image")


ax2.set_title("annotated image")
ax2.add_patch(plt.Rectangle((10, 10), 20, 20, linewidth=1, edgecolor='r', facecolor='none'))
ax2.imshow(image_converstion)

# Optional: Hide axis ticks for clean look
ax2.axis('off')
ax1.axis('off')

# Show the image
plt.show()

# save the image 
image_converstion.save("random_image.png")
x=np.linspace(0,2,100)
y=np.sin(x*2*np.pi) # we mulit
plt.plot(x,y)

plt.show()



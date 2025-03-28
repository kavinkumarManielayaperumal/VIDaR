import matplotlib.pyplot as plt
import numpy as np

# Create a random RGB image (100x100 pixels, 3 channels)
image = np.random.rand(100, 100, 3)

# Create a figure and axis
fig, (ax1,ax2) = plt.subplots(1,2) 

fig.suptitle('random image vs annotated image', fontsize=20)

ax1.imshow(image)
ax1.set_title("orginal random image")


ax2.set_title("annotated image")
ax2.add_patch(plt.Rectangle((10, 10), 20, 20, linewidth=1, edgecolor='r', facecolor='none'))
ax2.imshow(image)

# Optional: Hide axis ticks for clean look
ax2.axis('off')
ax1.axis('off')

# Show the image
plt.show()


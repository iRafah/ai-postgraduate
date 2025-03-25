import cv2
import matplotlib.pyplot as plt

# Load image.
image = cv2.imread('bird.jpg')

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_soft = cv2.GaussianBlur(image, (15, 25), 0)

plt.imshow(image_rgb)
plt.axis('off') # disable axis
plt.show()

# Show image.
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Image converted to gray
plt.imshow(image_gray, cmap='gray')
plt.axis('off') # disable axis
plt.show()

# Resize the image
width = 800
height = 600
dimension = (width, height)
resized_image = cv2.resize(image, dimension, 
                           interpolation=cv2.INTER_AREA)

# Convert an image from RGB to GBR
resized_image_rgb = cv2.cvtColor(resized_image,
                                 cv2.COLOR_BGR2RGB)
soft_image_rgb = cv2.cvtColor(image_soft,
                                 cv2.COLOR_BGR2RGB)

plt.imshow(resized_image_rgb)
plt.axis('off') # disable axis
plt.show()

plt.imshow(soft_image_rgb)
plt.axis('off') # disable axis
plt.show()
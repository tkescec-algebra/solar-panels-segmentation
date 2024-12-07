import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '../datasets/dataset/test/PV01_325123_1204235.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the results
plt.figure(figsize=(15, 10))
# Original image
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()
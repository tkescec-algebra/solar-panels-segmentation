import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brent

# Load the image
image_path = '../../datasets/dataset/test/PV01_325239_1203816.jpg'  # Replace with your image path
image_bgr = cv2.imread(image_path)

if image_bgr is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Convert BGR to RGB for visualization
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert to grayscale for Sobel operations
gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Apply Gaussian smoothing (optional)
blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

# Apply Sobel filters
sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobelx, sobely)

# Normalize Sobel channels to [0, 255]
sobelx_norm = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX)
sobely_norm = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX)
sobel_combined_norm = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)

# Scale Sobel channels to [0, 1]
sobelx_scaled = sobelx_norm.astype(np.float32) / 255.0
sobely_scaled = sobely_norm.astype(np.float32) / 255.0
sobel_combined_scaled = sobel_combined_norm.astype(np.float32) / 255.0

# Expand dimensions to match (H, W, 1)
sobelx_exp = np.expand_dims(sobelx_scaled, axis=2)
sobely_exp = np.expand_dims(sobely_scaled, axis=2)
sobel_combined_exp = np.expand_dims(sobel_combined_scaled, axis=2)

# Combine original RGB image with Sobel channels
# Convert original RGB image to [0, 1]
image_rgb_scaled = image_rgb.astype(np.float32) / 255.0

# Concatenate along the channel axis
combined = np.concatenate([image_rgb_scaled, sobelx_exp, sobely_exp, sobel_combined_exp], axis=2)  # (H, W, 6)

# Prikaz pojedinačnih kanala
plt.figure(figsize=(15, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)  # 2x3 grid za 6 kanala
    if i < 3:
        # Original RGB kanali
        plt.imshow(combined[:, :, i])
        plt.title(f'RGB Channel {i+1}')
    else:
        # Sobel kanali
        plt.imshow(combined[:, :, i], cmap='gray')
        if i == 3:
            plt.title('Sobel X')
        elif i == 4:
            plt.title('Sobel Y')
        elif i == 5:
            plt.title('Sobel Combined')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Kreiranje pseudo-RGB slike koristeći Sobel kanale
pseudo_rgb = np.zeros_like(image_rgb_scaled)  # Napravi praznu sliku istih dimenzija

# Mapiranje Sobel kanala na RGB kanale
pseudo_rgb[:, :, 0] = combined[:, :, 3]  # Sobel X za crvenu
pseudo_rgb[:, :, 1] = combined[:, :, 4]  # Sobel Y za zelenu
pseudo_rgb[:, :, 2] = combined[:, :, 5]  # Sobel Combined za plavu

plt.figure(figsize=(8, 6))
plt.imshow(pseudo_rgb)
plt.title("Pseudo-RGB Image")
plt.axis('off')
plt.show()

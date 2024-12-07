import cv2
import numpy as np

def sobel_edge_detection(image):
    """
    Primjena Sobelovog detekcije ivica na RGB slici.

    Args:
        image (numpy.ndarray): Ulazna RGB slika u obliku (H, W, 3).

    Returns:
        List[numpy.ndarray]: Lista od tri grayscale slike sa Sobelovim ivicama (sobelx, sobely, sobel_combined).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred_image = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel u x, y i kombinirani
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)

    # Normalizacija na [0, 255]
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    sobely = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    # Skaliranje na [0, 1]
    sobelx /= 255.0
    sobely /= 255.0
    sobel_combined /= 255.0

    return [
        np.expand_dims(sobelx, axis=2),
        np.expand_dims(sobely, axis=2),
        np.expand_dims(sobel_combined, axis=2)
    ]

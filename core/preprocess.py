import cv2
import numpy as np  

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    apply basic image processing pipeline
    """
    # Convert to grayscale : reduces complexity (3 channels to 1), removes color noise 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur (denoising) : smooths noise, helps detection stability (camera noise removal)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # histogram equalization (contrast enhancement) : improves contrast, makes dark objectives visible (very important in low-light conditions - industrial cameras)
    equalized = cv2.equalizeHist(blurred) 

    # Perform edge detection (canny)
    processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)  # convert back to 3 channels for edge detection

    return processed
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def preprocess_pipeline(image):
    results = {}

    # 1. Gaussian blur
    blur = cv2.GaussianBlur(image, (5,5), 0)
    results["blur"] = blur

    # 2. Histogram equalization
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    results["equalized"] = equalized

    # 3. Sobel (derivatives)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobelx = cv2.convertScaleAbs(sobelx)
    sobelx = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
    results["sobel"] = sobelx

    # 4. Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    results["laplacian"] = laplacian

    # 5. Morphological operations
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    dilation = cv2.cvtColor(dilation, cv2.COLOR_GRAY2BGR)
    results["dilation"] = dilation

    return results


def run_detection(image):
    results = model(image)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        detections.append({
            "class": model.names[cls],
            "confidence": conf
        })

    return detections

def summarize_detections(detections):
    counts = len(detections)
    avg_conf = sum(d["confidence"] for d in detections) / counts if counts > 0 else 0
    return counts, avg_conf
# Load image
image = cv2.imread("/Users/burakberkerbasergun/Desktop/objectDetection/TMNT-multi-cardboard-cutout-pack-buy-now-at-starstills__99828.jpg")

# Original detection
original_det = run_detection(image)

# Processed versions
processed = preprocess_pipeline(image)

print("Original:", original_det)

for name, img in processed.items():
    det = run_detection(img)
    print(f"{name}:", det)

    cv2.imshow(name, img)

counts, avg_conf = summarize_detections(original_det)
print(f"Summary - Count: {counts}, Avg Confidence: {avg_conf:.2f}")
cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

combined = np.hstack([
    image,
    processed["equalized"],
    processed["laplacian"]
])

cv2.imshow("Comparison", combined)
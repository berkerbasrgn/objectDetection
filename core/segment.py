import cv2
import numpy as np
from ultralytics import YOLO

seg_model = YOLO("yolov8n-seg.pt")


def segment_otsu(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def segment_watershed(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    result = image.copy()
    result[markers == -1] = [0, 0, 255]
    return result


def segment_yolo(image: np.ndarray) -> tuple:
    results = seg_model(image)
    overlay = image.copy()
    detections = []

    if results[0].masks is None:
        return overlay, detections

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = seg_model.names[cls]

        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        binary_mask = (mask_resized > 0.5).astype(np.uint8)

        color = np.random.randint(50, 200, size=3).tolist()
        colored = np.zeros_like(image)
        colored[:] = color
        overlay = np.where(
            binary_mask[:, :, None] == 1,
            cv2.addWeighted(overlay, 0.6, colored, 0.4, 0),
            overlay
        )

        xyxy = box.xyxy[0].tolist()
        x1, y1 = int(xyxy[0]), int(xyxy[1])
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({"class": label, "confidence": conf, "bbox": xyxy})

    return overlay, detections

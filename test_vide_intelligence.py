import cv2
import numpy as np
from ultralytics import YOLO
import os 

output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")
#Screen Recording 2026-05-10 at 22.08.56.mov
video_path = "Screen Recording 2026-05-10 at 22.08.56.mov"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(" ERROR: Video cannot be opened")
    exit()

# Video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    "output_video.mp4",
    fourcc,
    20.0,
    (width * 2, height * 2)
)

# Preprocessing
def preprocess_pipeline(image):
    results = {}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalization
    eq = cv2.equalizeHist(gray)
    eq = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    results["equalized"] = eq

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    cl = cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR)
    results["clahe"] = cl

    # Gaussian Blur
    results["gaussian_blur"] = cv2.GaussianBlur(image, (5,5), 0)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    results["sharpen"] = cv2.filter2D(image, -1, kernel)

    # Sobel
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel = cv2.convertScaleAbs(sobel)
    results["sobel"] = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    # Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    results["laplacian"] = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    # Dilation
    kernel = np.ones((5,5), np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=1)
    results["dilation"] = cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)

    return results


# Detection
def run_detection_with_boxes(image):

    results = model(image)

    detections = []

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[cls],
            "confidence": conf,
            "bbox": xyxy
        })

    return detections


# Drawing

def draw_detections(image, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        if "track_id" in det:
            label = f"ID:{det['track_id']} {det['class']} {det['confidence']:.2f}"
        else:
            label = f"{det['class']} {det['confidence']:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)
    return image




# Metrics

def summarize(detections):
    count = len(detections)
    avg_conf = sum(d["confidence"] for d in detections) / count if count > 0 else 0
    return count, avg_conf

# ORB Feature Descriptor

def extract_orb_features(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=300)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    feature_img = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0,255,0),
        flags=0
    )

    return feature_img, len(keypoints)

# Motion Features - Optical Flow

def draw_optical_flow(prev_gray, curr_gray, frame):

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    step = 20

    h, w = curr_gray.shape

    for y in range(0, h, step):
        for x in range(0, w, step):

            fx, fy = flow[y, x]

            end_x = int(x + fx)
            end_y = int(y + fy)

            cv2.arrowedLine(
                frame,
                (x, y),
                (end_x, end_y),
                (0, 0, 255),
                1,
                tipLength=0.3
            )

    return frame



# Stats

stats = {
    "original": [],
    "equalized": [],
    "clahe": [],
    "gaussian_blur": [],
    "sharpen": [],
    "sobel": [],
    "laplacian": [],
    "dilation": []
}


# Main Loop

frame_count = 0
id_map = {}
next_display_id = 1
# Optical Flow variables
prev_gray = None
while frame_count < 100:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Frame read failed, stopping")
        break

    # TRACKING
    track_results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    orig_det = []

    for box in track_results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        display_id = -1

        if box.id is not None:

            real_id = int(box.id[0])

            # Assign ONLY ONCE
            if real_id not in id_map:
                id_map[real_id] = next_display_id
                next_display_id += 1

            display_id = id_map[real_id]

        orig_det.append({
            "class": model.names[cls],
            "confidence": conf,
            "bbox": xyxy,
            "track_id": display_id
        })

    stats["original"].append(summarize(orig_det))

    frame_with_boxes = draw_detections(frame.copy(), orig_det)
    # MOTION FEATURES (Optical Flow)

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    motion_frame = frame.copy()

    if prev_gray is not None:

        motion_frame = draw_optical_flow(
            prev_gray,
            curr_gray,
            motion_frame
        )

    prev_gray = curr_gray
    
    
    # ORB Features
    orb_frame, orb_count = extract_orb_features(frame)

    cv2.putText(
        orb_frame,
        f"ORB Features: {orb_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # PREPROCESSING
    processed = preprocess_pipeline(frame)

    for name, img in processed.items():
        det = run_detection_with_boxes(img)
        stats[name].append(summarize(det))

    # Visualization
    eq_det = run_detection_with_boxes(processed["equalized"])
    eq_frame = draw_detections(processed["equalized"].copy(), eq_det)

    top_row = np.hstack([frame_with_boxes, eq_frame])
    bottom_row = np.hstack([orb_frame, motion_frame])
    combined = np.vstack([top_row, bottom_row])
    out.write(combined)
    cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", combined)

    frame_count += 1


cap.release()
out.release()

print("\n VIDEO RESULTS SUMMARY ")

for method, values in stats.items():
    if len(values) == 0:
        continue

    avg_count = np.mean([v[0] for v in values])
    avg_conf = np.mean([v[1] for v in values])

    print(f"{method}: avg_count={avg_count:.2f}, avg_conf={avg_conf:.2f}")
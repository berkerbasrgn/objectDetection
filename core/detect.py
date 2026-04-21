from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

def extract_detections(results, conf_threshold=0.5):
    detections = []
    
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue    
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        detections.append({
            "class": model.names[cls],
            "confidence": conf,
            "bbox": xyxy
        })
    return {"detections": detections}

def run_detection(image, conf_threshold=0.5):
    results = model(image)
    return extract_detections(results, conf_threshold)
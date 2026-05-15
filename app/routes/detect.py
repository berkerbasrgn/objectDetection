from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
from PIL import Image
from core.detect import extract_detections, run_detection
from core.preprocess import preprocess_image
from core.track import SimpleTracker
from core.decision import IntelligenceEngine
from core.video_pipeline import process_video
from core.segment import segment_otsu, segment_watershed, segment_yolo
tracker = SimpleTracker()
router = APIRouter()
engine = IntelligenceEngine()


@router.post("/detect")

async def detect(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)
    #raw detection
    raw_detections = run_detection(image_np)['detections']
    # preprocessed detection

    tracked = tracker.update(raw_detections)
    insights = engine.update(tracked)
    return {
        "tracked_objects": tracked,
        "insights": insights
    }

@router.post("/segment")
async def segment(file: UploadFile = File(...), method: str = "yolo"):
    image = Image.open(file.file).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if method == "otsu":
        result = segment_otsu(image_np)
        detections = []
    elif method == "watershed":
        result = segment_watershed(image_np)
        detections = []
    else:
        result, detections = segment_yolo(image_np)

    _, buffer = cv2.imencode(".jpg", result)
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"X-Detections": str(len(detections))}
    )


@router.post("/process_video")
async def process_video_endpoint(file: UploadFile = File(...)):
    temp_path = f"/tmp/{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(await file.read())

    results = process_video(temp_path)

    return {
        "frames_processed": len(results),
        "sample_output": results[:2]  # return first 2 frames' results as sample
    }
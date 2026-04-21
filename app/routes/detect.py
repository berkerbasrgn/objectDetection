from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from PIL import Image
from core.detect import extract_detections, run_detection
from core.preprocess import preprocess_image

router = APIRouter()

@router.post("/detect")
async def detect(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)
    #raw detection
    raw_detections = run_detection(image_np)
    # preprocessed detection

    preprocessed = preprocess_image(image_np)
    processed_detections = run_detection(preprocessed)

    return {
        "raw_detections": raw_detections["detections"],
        "processed_detections": processed_detections["detections"]
    }


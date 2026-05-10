import cv2
from core.detect import run_detection
from core.track import SimpleTracker
from core.decision import IntelligenceEngine

def process_video(video_path):
    tracker = SimpleTracker()
    engine = IntelligenceEngine()

    cap = cv2.VideoCapture(video_path)

    results_summary = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_detection(frame)["detections"]
        tracked = tracker.update(detections)
        insights = engine.update(tracked)

        results_summary.append({
            "tracked": tracked,
            "insights": insights
        })

    cap.release()

    return results_summary
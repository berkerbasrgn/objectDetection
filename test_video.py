import cv2
from core.detect import run_detection
from core.track import SimpleTracker

tracker = SimpleTracker()

cap = cv2.VideoCapture(0)  # webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = run_detection(frame)["detections"]
    tracked = tracker.update(detections)

    for obj in tracked:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        obj_id = obj["id"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
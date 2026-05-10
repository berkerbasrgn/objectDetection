import numpy as np

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def _get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def update(self, detections):
        new_objects = {}

        for det in detections:
            centroid = self._get_centroid(det["bbox"])

            matched_id = None
            min_dist = float("inf")

            for obj_id, old_centroid in self.objects.items():
                dist = np.linalg.norm(centroid - old_centroid)

                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is None:
                matched_id = self.next_id
                self.next_id += 1

            new_objects[matched_id] = centroid
            det["id"] = matched_id

        self.objects = new_objects
        return detections
# counts people
# tracks object lifetime
# creates alerts

class IntelligenceEngine:
    def __init__(self):
        self.object_history = {}  # id : frames seen

    def update(self, tracked_objects):
        events = []
        person_count = 0

        for obj in tracked_objects:
            obj_id = obj["id"]
            obj_class = obj["class"]

            # count people
            if obj_class == "person":
                person_count += 1

            # track duration
            if obj_id not in self.object_history:
                self.object_history[obj_id] = 0

            self.object_history[obj_id] += 1

            # if object seen for long time then event
            if self.object_history[obj_id] > 30:
                events.append(f"object_{obj_id}_stationary")

        # crowd detection
        if person_count >= 3:
            events.append("crowd_detected")

        return {
            "person_count": person_count,
            "events": events
        }
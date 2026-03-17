import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Sort


class PeopleCounter:

    def __init__(self):

        self.model = YOLO("yolov8n.pt")

        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        self.total_count = 0
        self.counted_ids = set()

        self.line_y = 350


    def process_frame(self, frame):

        results = self.model(frame)[0]

        detections = []

        for box in results.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 0 and conf > 0.4:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append([x1, y1, x2, y2, conf])

        detections = np.array(detections)

        if len(detections) == 0:
            tracks = []
        else:
            tracks = self.tracker.update(detections)

        for track in tracks:

            x1, y1, x2, y2, track_id = map(int, track)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            if cy > self.line_y and track_id not in self.counted_ids:

                self.total_count += 1
                self.counted_ids.add(track_id)

        cv2.line(frame,(0,self.line_y),(frame.shape[1],self.line_y),(0,0,255),3)

        cv2.putText(
            frame,
            f"Count: {self.total_count}",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

        return frame
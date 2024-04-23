"""

12/03/2024 20:04
Eduardo Bitencourt

"""

from sort import Sort
import numpy as np
import cv2
from ultralytics import YOLO

my_model = YOLO("yolov8n.pt")

video = cv2.VideoCapture("gado.mp4")

tracker = Sort(max_age=20, min_hits=5, iou_threshold=0.3)

while True:

    ret, img = video.read()

    if ret:

        results = my_model(img, verbose=False)

        detections = np.empty((0,5))
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == "cow":
                    x1, y1, x2, y2 = int(box.xyxy[0,0]), int(box.xyxy[0,1]), int(box.xyxy[0,2]), int(box.xyxy[0,3])
                    detections = np.append(detections, np.array([x1, y1, x2, y2, float(box.conf[0])]).reshape(1,5), axis=0)

        tracked_objects = tracker.update(detections)

        for tracked_object in tracked_objects:
            x1, y1, x2, y2 = int(tracked_object[0]), int(tracked_object[1]), int(tracked_object[2]), int(tracked_object[3])
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 3)

            annotation = f'cow id {int(tracked_object[4])}'
            (text_w, text_h), _ = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img = cv2.rectangle(img, (x1, y1 - 20), (x1 + text_w, y1), (255,0,0), -1)
            img = cv2.putText(img, annotation, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # img = cv2.



        cv2.imshow("Image", img)
        cv2.waitKey(1)

    else:
        break

cv2.destroyAllWindows()

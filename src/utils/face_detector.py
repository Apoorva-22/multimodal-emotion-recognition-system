import cv2
from ultralytics import YOLO

face_model = YOLO("checkpoints/yolov8-face.pt")

def detect_face(frame):
    results = face_model(frame, verbose=False)

    best_face = None
    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)

            area = (x2 - x1) * (y2 - y1)

            if area > max_area:
                max_area = area
                best_box = (x1, y1, x2, y2)
                best_face = frame[y1:y2, x1:x2]

    return best_face, best_box
import cv2
from src.utils.face_detector import detect_face

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
last_box = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # run YOLO every 10 frames
    if frame_count % 10 == 0:
        face, box = detect_face(frame)
        if box:
            last_box = box

    if last_box:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
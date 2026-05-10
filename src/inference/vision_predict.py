import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from src.models.vision_model import VisionEmotionModel
from src.utils.face_detector import detect_face

# emotion labels
emotion_labels = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Neutral",
    4: "Sad"
}

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained emotion model
model = VisionEmotionModel().to(device)
model.load_state_dict(torch.load("checkpoints/vision.pt", map_location=device))
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
last_box = None
last_emotion = "Detecting..."

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # detect every 10 frames
    if frame_count % 10 == 0:
        face, box = detect_face(frame)

        if face is not None and box is not None:
            last_box = box

            # preprocess face
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)

            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output, _ = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            last_emotion = emotion_labels[pred]

    if last_box:
        x1, y1, x2, y2 = last_box

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(
            frame,
            last_emotion,
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
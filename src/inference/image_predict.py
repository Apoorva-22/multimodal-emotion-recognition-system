import cv2
import torch
import numpy as np
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from src.models.vision_model import VisionEmotionModel
from src.utils.face_detector import detect_face

emotion_labels = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


def predict_image_emotion(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        raise ValueError("Invalid image path or image could not be loaded")

    face, box = detect_face(frame)

    if face is None:
        face = frame

    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(face_rgb)

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(input_tensor)

        probs = F.softmax(
            outputs,
            dim=1
        ).cpu().numpy()[0]

        pred_idx = np.argmax(probs)

    return {
        "emotion": emotion_labels[pred_idx],
        "probabilities": probs.tolist()
    }

import torch
from sklearn.metrics import classification_report
from src.models.vision_model import VisionEmotionModel
from src.data.datasets import get_video_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, test_loader = get_video_dataloaders(
    "data/processed/video/train",
    "data/processed/video/test"
)

model = VisionEmotionModel().to(device)
model.load_state_dict(torch.load("checkpoints/vision.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, _ = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.vision_model import VisionEmotionModel
from src.data.datasets import get_video_dataloaders

train_loader, test_loader = get_video_dataloaders(
    "data/processed/video/train",
    "data/processed/video/test"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VisionEmotionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/vision.pt")
print("Vision model saved")
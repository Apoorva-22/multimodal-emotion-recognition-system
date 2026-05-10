import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class VisionEmotionModel(nn.Module):
    def __init__(self):
        super(VisionEmotionModel, self).__init__()

        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            256
        )

        self.classifier = nn.Linear(256, 5)

    def forward(self, x):
        embedding = self.backbone(x)
        output = self.classifier(embedding)
        return output, embedding
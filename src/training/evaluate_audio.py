import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from src.models.audio_model import AudioEmotionModel

# load data
X = np.load("data/processed/audio/X.npy")
y = np.load("data/processed/audio/y.npy")

# encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioEmotionModel().to(device)
model.load_state_dict(
    torch.load("checkpoints/audio.pt", map_location=device)
)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    outputs, _ = model(X_test.to(device))
    preds = torch.argmax(outputs, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(y_test.cpu().numpy())

print(classification_report(all_labels, all_preds))
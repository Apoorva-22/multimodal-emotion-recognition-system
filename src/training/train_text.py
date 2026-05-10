import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader

from src.models.text_model import TextEmotionModel


# load data
df = pd.read_csv("data/processed/text/processed_text.csv")

texts = df["text"].tolist()
labels = df["emotion"].tolist()

# encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

print("Label mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(i, "->", label)

# tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors="pt"
)

input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]

labels = torch.tensor(labels)

# split
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    input_ids,
    attention_mask,
    labels,
    test_size=0.2,
    random_state=42
)

train_dataset = TensorDataset(
    X_train_ids,
    X_train_mask,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TextEmotionModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for ids, mask, labels in train_loader:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, _ = model(ids, mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={total_loss:.4f}")

torch.save(
    model.state_dict(),
    "checkpoints/text.pt"
)

print("Text model saved")
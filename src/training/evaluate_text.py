import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from transformers import DistilBertTokenizer

from src.models.text_model import TextEmotionModel


# load processed data
df = pd.read_csv(
    "data/processed/text/processed_text.csv"
)

texts = df["text"].tolist()
labels = df["emotion"].tolist()

# encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

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

labels_tensor = torch.tensor(labels_encoded)

# split same as training
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    input_ids,
    attention_mask,
    labels_tensor,
    test_size=0.2,
    random_state=42
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TextEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "checkpoints/text.pt",
        map_location=device
    )
)

model.eval()

with torch.no_grad():
    outputs, _ = model(
        X_test_ids.to(device),
        X_test_mask.to(device)
    )

    preds = torch.argmax(
        F.softmax(outputs, dim=1),
        dim=1
    ).cpu()

print(
    "Accuracy:",
    accuracy_score(y_test, preds)
)

print(
    classification_report(
        y_test,
        preds,
        target_names=label_encoder.classes_
    )
)
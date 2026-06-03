import torch
import torch.nn.functional as F
import numpy as np

from transformers import DistilBertTokenizer
from src.models.text_model import TextEmotionModel

tokenizer = None
model = None

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



def predict_text_emotion(text):
    
    tokenizer, model = load_text_model()
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=64,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs, _ = model(
            input_ids,
            attention_mask
        )

        probs = F.softmax(
            outputs,
            dim=1
        ).cpu().numpy()[0]

        pred_idx = np.argmax(probs)

    return {
        "emotion": emotion_labels[pred_idx],
        "probabilities": probs.tolist()
    }

def load_text_model():

    global tokenizer
    global model

    if tokenizer is None:

        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )

    if model is None:

        model = TextEmotionModel().to(device)

        model.load_state_dict(
            torch.load(
                "checkpoints/text.pt",
                map_location=device
            )
        )

        model.eval()

    return tokenizer, model
    

if __name__ == "__main__":
    sample_text = "I am feeling extremely happy today"

    result = predict_text_emotion(sample_text)

    print(result)

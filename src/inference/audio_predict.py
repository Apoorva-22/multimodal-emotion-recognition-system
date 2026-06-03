import librosa
import numpy as np
import torch
import torch.nn.functional as F

from src.models.audio_model import AudioEmotionModel

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

model = None

def preprocess_audio(audio_path):
    signal, sr = librosa.load(
        audio_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    mfcc = mfcc.T

    max_len = 100

    if mfcc.shape[0] < max_len:
        pad_width = max_len - mfcc.shape[0]

        mfcc = np.pad(
            mfcc,
            ((0, pad_width), (0, 0)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:max_len]

    return mfcc


def predict_audio_emotion(audio_path):
    model = load_vision_model()
    features = preprocess_audio(audio_path)

    features = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(features)

        probs = F.softmax(
            outputs,
            dim=1
        ).cpu().numpy()[0]

        pred_idx = np.argmax(probs)

    return {
        "emotion": emotion_labels[pred_idx],
        "probabilities": probs.tolist()
    }

def load_vision_model():

    global model

    if model is None:

        model = VisionEmotionModel().to(device)

        model.load_state_dict(
            torch.load(
                "checkpoints/vision.pt",
                map_location=device
            )
        )

        model.eval()

    return model
if __name__ == "__main__":
    result = predict_audio_emotion(
        "sample.wav"
    )

    print(result)

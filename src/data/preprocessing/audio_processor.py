import os
import librosa
import numpy as np

DATA_PATH = "data/raw/ravdess/audio_speech_actors_01-24"
SAVE_PATH = "data/processed/audio"

os.makedirs(SAVE_PATH, exist_ok=True)

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear"
}

X = []
y = []

for root, dirs, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_code = file.split("-")[2]

            if emotion_code in emotion_map:
                file_path = os.path.join(root, file)

                signal, sr = librosa.load(file_path, sr=22050)

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

                X.append(mfcc)
                y.append(emotion_map[emotion_code])

print("Total samples:", len(X))

np.save(
    os.path.join(SAVE_PATH, "X.npy"),
    np.array(X)
)

np.save(
    os.path.join(SAVE_PATH, "y.npy"),
    np.array(y)
)

print("Audio preprocessing complete")
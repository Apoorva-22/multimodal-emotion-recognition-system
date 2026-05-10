import numpy as np

emotion_labels = [
    "angry",
    "fear",
    "happy",
    "neutral",
    "sad"
]

def fuse_predictions(vision_probs, audio_probs, text_probs):
    vision_probs = np.array(vision_probs)
    audio_probs = np.array(audio_probs)
    text_probs = np.array(text_probs)

    vision_weight = 0.4
    audio_weight = 0.3
    text_weight = 0.3

    # boost highly confident modality
    if np.max(vision_probs) > 0.95:
        vision_weight = 0.6
        audio_weight = 0.2
        text_weight = 0.2

    final_probs = (
        vision_weight * vision_probs +
        audio_weight * audio_probs +
        text_weight * text_probs
    )

    final_emotion = emotion_labels[np.argmax(final_probs)]

    return final_emotion, final_probs.tolist()



    
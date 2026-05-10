from src.inference.audio_predict import predict_audio_emotion
from src.inference.text_predict import predict_text_emotion
from src.models.fusion_model import fuse_predictions
from src.inference.image_predict import predict_image_emotion

def predict_multimodal(image_path, audio_path, text_input):

    vision_result = predict_image_emotion(image_path)
    audio_result = predict_audio_emotion(audio_path)
    text_result = predict_text_emotion(text_input)

    final_emotion, final_probs = fuse_predictions(
        vision_probs=vision_result["probabilities"],
        audio_probs=audio_result["probabilities"],
        text_probs=text_result["probabilities"]
    )

    return {
        "vision_prediction": vision_result,
        "audio_prediction": audio_result,
        "text_prediction": text_result,
        "final_emotion": final_emotion,
        "final_probabilities": final_probs
    }


if __name__ == "__main__":
    result = predict_multimodal(
        audio_path="sample.wav",
        text_input="I am feeling happy today"
    )

    print(result)
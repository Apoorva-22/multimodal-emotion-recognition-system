from huggingface_hub import hf_hub_download
import os

REPO_ID = "apoorva6627/multimodal-emotion-models"

MODELS = [
    "audio.pt",
    "vision.pt",
    "text.pt",
    "yolov8-face.pt"
]


def ensure_models():

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    for model_name in MODELS:

        local_path = os.path.join(
            "checkpoints",
            model_name
        )

        if not os.path.exists(local_path):

            print(
                f"Downloading {model_name}..."
            )

            hf_hub_download(
                repo_id=REPO_ID,
                filename=model_name,
                local_dir="checkpoints"
            )

    print("All models ready")

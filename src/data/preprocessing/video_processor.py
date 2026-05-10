import os
import shutil

VALID_CLASSES = ["angry", "happy", "sad", "neutral", "fear"]

splits = ["train", "test"]

for split in splits:
    source_dir = f"data/raw/fer2013/{split}"
    target_dir = f"data/processed/video/{split}"

    os.makedirs(target_dir, exist_ok=True)

    for emotion in VALID_CLASSES:
        source_path = os.path.join(source_dir, emotion)
        target_path = os.path.join(target_dir, emotion)

        if os.path.exists(source_path):
            shutil.copytree(
                source_path,
                target_path,
                dirs_exist_ok=True
            )
            print(f"Copied {emotion} from {split}")

print("Processing complete")
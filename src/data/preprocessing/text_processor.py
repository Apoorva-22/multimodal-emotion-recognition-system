import pandas as pd
import os

DATA_PATH = "data/raw/goemotions/data/train.tsv"
SAVE_PATH = "data/processed/text"

os.makedirs(SAVE_PATH, exist_ok=True)

# GoEmotions emotion index mapping
target_emotions = {
    13: "fear",
    17: "joy",
    25: "neutral",
    26: "sadness",
    2: "anger"
}

rows = []

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["text", "labels", "id"]
)

for _, row in df.iterrows():
    labels = row["labels"].split(",")

    matched_labels = []

    for label in labels:
        label = int(label)

        if label in target_emotions:
            mapped = target_emotions[label]

            if mapped == "joy":
                mapped = "happy"
            elif mapped == "sadness":
                mapped = "sad"

            matched_labels.append(mapped)

    # only keep samples with exactly one valid target label
    if len(matched_labels) == 1:
        rows.append({
            "text": row["text"],
            "emotion": matched_labels[0]
        })

processed_df = pd.DataFrame(rows)

print(processed_df["emotion"].value_counts())

processed_df.to_csv(
    os.path.join(SAVE_PATH, "processed_text.csv"),
    index=False
)

print("Text preprocessing complete")
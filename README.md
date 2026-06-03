# Multimodal Emotion Recognition System

A production-style multimodal AI system that predicts human emotions using:

- Facial expressions (Vision)
- Speech tone (Audio)
- Text sentiment (NLP)

Final emotions predicted:

- Happy
- Sad
- Angry
- Neutral
- Fear

---

## Architecture

### Vision Pipeline
Image/Webcam Frame  
→ YOLO Face Detection  
→ ResNet18 Emotion Classifier  
→ Emotion Probabilities

Dataset: FER2013

Accuracy: 61%

---

## Audio Pipeline
Audio Input (.wav)  
→ MFCC Feature Extraction (Librosa)  
→ LSTM Model  
→ Emotion Probabilities

Dataset: RAVDESS

Accuracy: 61%

---

## Text Pipeline
Text Input  
→ DistilBERT Tokenization  
→ DistilBERT Classifier  
→ Emotion Probabilities

Dataset: GoEmotions

Accuracy: 82%

---

## Fusion Pipeline

Emotion probabilities from all modalities are fused using:

### Confidence-Aware Late Fusion

- Default weights:
  - Vision → 0.4
  - Audio → 0.3
  - Text → 0.3

- If any modality confidence >95%, its weight is boosted.

Final output:
- Final emotion
- Final probability distribution

---

# Tech Stack

- Python
- PyTorch
- FastAPI
- OpenCV
- YOLO
- Librosa
- HuggingFace Transformers
- NumPy
- Scikit-learn

---

# Folder Structure

emotion-recognition-system/

src/
- models/
- training/
- inference/
- api/
- data/

checkpoints/

notebooks/

---

# API Endpoint

POST:

/predict-emotion

Inputs:

- image
- audio
- text

Returns:

- vision prediction
- audio prediction
- text prediction
- final fused emotion

---

# Results

| Modality | Accuracy |
|----------|------------|
| Vision | 61% |
| Audio | 61% |
| Text | 82% |

---

# Features

- Real-time webcam detection
- Image upload support
- Audio upload support
- Text input support
- FastAPI backend
- Multimodal fusion

---

# Future Improvements

- Better text generalization
- Video upload support
- Trainable fusion model
- Docker deployment
- Frontend UI

---

# How to Run

```bash
pip install -r requirements.txt
```

Train models:

```bash
python -m src.training.train_vision
python -m src.training.train_audio
python -m src.training.train_text
```

Run API:

```bash
uvicorn src.api.main:app --reload
```

Open:

```plaintext
http://127.0.0.1:8000/docs
```

---

## Demo

### FastAPI Backend API
![FastAPI](screenshots/fastapi.png)

### Streamlit Frontend
![Frontend](screenshots/frontend_input.png)

### Prediction Results
![Results](screenshots/frontend_results.png)

# Real World Applications

- Mental health assistants
- Customer support analysis
- Interview analytics
- Smart education systems
- Healthcare monitoring

# NOTE
- Pretrained model checkpoints excluded due to GitHub size limits.

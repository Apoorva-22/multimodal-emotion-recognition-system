import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000/predict-emotion"

st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="😊",
    layout="wide"
)

st.title("😊 Multimodal Emotion Recognition System")
st.markdown(
    "Upload **image + audio + text** to predict final emotion using multimodal AI."
)

# sidebar
st.sidebar.header("About")
st.sidebar.write("""
Models Used:
- Vision → YOLO + ResNet18
- Audio → LSTM + MFCC
- Text → DistilBERT
- Fusion → Confidence-aware Late Fusion
""")

# inputs
col1, col2 = st.columns(2)

with col1:
    image_file = st.file_uploader(
        "Upload Face Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        st.image(image_file, caption="Uploaded Image")

with col2:
    audio_file = st.file_uploader(
        "Upload Audio File",
        type=["wav"]
    )
    if audio_file:
        st.audio(audio_file)

text_input = st.text_area(
    "Enter Text",
    placeholder="Type your emotion here..."
)

if st.button("Predict Emotion"):

    if image_file and audio_file and text_input:

        with st.spinner("Analyzing emotions..."):

            files = {
                "image": image_file,
                "audio": audio_file
            }

            data = {
                "text": text_input
            }

            response = requests.post(
                API_URL,
                files=files,
                data=data
            )

            result = response.json()

        st.markdown(
            f"""
            ## 🎯 Final Emotion: **{result['final_emotion'].upper()}**
            """
        )

        # individual predictions
        st.subheader("Modality Predictions")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Vision",
                result["vision_prediction"]["emotion"]
            )

        with c2:
            st.metric(
                "Audio",
                result["audio_prediction"]["emotion"]
            )

        with c3:
            st.metric(
                "Text",
                result["text_prediction"]["emotion"]
            )

        # final probabilities chart
        st.subheader("Final Emotion Probabilities")

        emotions = [
            "Angry",
            "Fear",
            "Happy",
            "Neutral",
            "Sad"
        ]

        probs = result["final_probabilities"]

        df = pd.DataFrame({
            "Emotion": emotions,
            "Probability": probs
        })

        fig = px.pie(
            df,
            names="Emotion",
            values="Probability",
            title="Emotion Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.error(
            "Please upload image, audio and text input."
        )
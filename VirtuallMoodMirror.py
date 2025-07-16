import streamlit as st
import numpy as np
import cv2
from deepface import DeepFace
import random

# Reactions dictionary
reactions = {
    "happy": ["😃 You look so happy! Here's a dancing cat GIF!", "😄 Keep smiling!"],
    "sad": ["😢 Why are you crying?", "🥺 Want a virtual hug? 🤗"],
    "angry": ["😡 Chill bro!", "🔥 Take a deep breath!"],
    "neutral": ["😐 Bruh...", "🫤 Feeling meh?"]
}

st.set_page_config(page_title="Virtual Mood Mirror", layout="centered")
st.title("🪞 Virtual Mood Mirror")
st.write("Take a photo and let me guess your mood!")

# Camera input
img_file_buffer = st.camera_input("📸 Take a photo")

if img_file_buffer is not None:
    # Convert image to OpenCV format
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    st.image(frame, caption="Your captured photo", channels="BGR")

    try:
        # Detect emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        st.subheader(f"🧠 Detected Emotion: `{emotion}`")

        # Show response
        for key in reactions:
            if key in emotion.lower():
                st.success(random.choice(reactions[key]))
    except Exception as e:
        st.error("🙁 Couldn't detect your face. Try again.")

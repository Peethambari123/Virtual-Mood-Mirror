import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import random

# Set page config
st.set_page_config(page_title="Virtual Mood Mirror", layout="centered")

# Emotion reactions
reactions = {
    "happy": ["😃 You look so happy! Here's a dancing cat GIF!", "😄 Keep smiling!"],
    "sad": ["😢 Why are you crying?", "🥺 Want a virtual hug? 🤗"],
    "angry": ["😡 Chill bro!", "🔥 Take a deep breath!"],
    "neutral": ["😐 Bruh...", "🫤 Feeling meh?"]
}

st.title("🪞 Virtual Mood Mirror")
st.write("Let me guess your mood... Take a selfie!")

# Capture image from webcam
img_file_buffer = st.camera_input("Take a photo")

if img_file_buffer is not None:
    # Convert image to numpy array
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)

    # Show the image
    st.image(frame, channels="BGR", caption="Your photo")

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        st.subheader(f"🧠 Detected Emotion: `{emotion}`")

        for key in reactions:
            if key in emotion.lower():
                st.success(random.choice(reactions[key]))
    except Exception as e:
        st.error("🙁 Face not detected properly. Try again!")


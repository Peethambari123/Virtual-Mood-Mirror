import cv2
from deepface import DeepFace
import random
import time

# Mapping emotions to memes or text reactions
reactions = {
    "happy": ["😃 You look so happy! Here's a dancing cat GIF!", "😄 Keep smiling!"] ,
    "sad": ["😢 Why are you crying?", "🥺 Want a virtual hug? 🤗"],
    "angry": ["😡 Chill bro!", "🔥 Take a deep breath!"],
    "neutral": ["😐 Bruh...", "🫤 Feeling meh?"]
}

# Start video capture
cap = cv2.VideoCapture(0)  # Works on PC/Laptop

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Show live feed
    cv2.imshow("Virtual Mood Mirror", frame)
    
    # Analyze mood every 5 seconds
    if int(time.time()) % 5 == 0:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            print(f"Detected Emotion: {emotion}")
            
            # Get reaction
            for key in reactions.keys():
                if key in emotion:
                    print(random.choice(reactions[key]))
        except:
            print("Face not detected!")
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

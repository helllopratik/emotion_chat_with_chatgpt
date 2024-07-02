import os
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
import numpy as np
import tensorflow as tf

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(face_image):
    model_emotion = tf.keras.models.load_model("emotiondetector.keras")
    if model_emotion is None:
        return None
    face_image = cv2.resize(face_image, (48, 48))
    img = extract_features(face_image)
    predictions = model_emotion.predict(img)
    emotion_index = np.argmax(predictions)
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

def process_video():
    webcam = cv2.VideoCapture(0)
    detected_emotion = None

    while True:
        ret, frame = webcam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_image = gray[y:y + h, x:x + w]
            if face_image.size == 0:
                continue
            detected_emotion = detect_emotion(face_image)
            if detected_emotion:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {detected_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        if detected_emotion:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    return detected_emotion

def get_response(question, detected_emotion):
    if detected_emotion == 'neutral':
        detected_emotion = 'peaceful'  # Treat neutral as peaceful

    response = chat.send_message(f" {detected_emotion}, {question}.")
    all_responses = []
    for part in response.parts:
        if part.text:
            all_responses.append(part.text)
    return " ".join(all_responses)

if __name__ == "__main__":
    chat = model.start_chat(history=[])
    question = input("Enter your question: ")
    detected_emotion = None
    while detected_emotion not in emotion_labels:
        detected_emotion = process_video()
    print(f"Detected Emotion: {detected_emotion}")
    response = get_response(question, detected_emotion)
    print(f"Bot: {response}")


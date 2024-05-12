# maincode.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
import numpy as np
import tensorflow as tf

# Load environment variables (assuming you have a GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the generative AI model using the API key
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')  # You can experiment with other models

chat = model.start_chat(history=[])

# Load the emotion detection model (replace with your trained model)
model_emotion = tf.keras.models.load_model("emotiondetector.keras")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (48, 48))
    img = extract_features(image)

    # Make predictions using your emotion detection model
    predictions = model_emotion.predict(img)
    emotion_index = np.argmax(predictions)  # Get the index of the highest probability
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

def process_video():
    webcam = cv2.VideoCapture(0)
    detected_emotion = None

    while True:
        ret, im = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        for (p, q, r, s) in faces:
            face_image = gray[q:q + s, p:p + r]
            detected_emotion = detect_emotion(face_image)
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(im, f"Emotion: {detected_emotion}", (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Emotion Detection", im)
        if detected_emotion:  # If emotion is detected, break the loop
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    return detected_emotion

while True:
    question = input("You: ")
    detected_emotion = process_video()
    print(f"Detected Emotion: {detected_emotion}")
    response = chat.send_message(f"Consider my emotion as {detected_emotion}, {question}")  # Get the response from the chat

    # Extract and print the text from the response
    all_responses = []
    for part in response.parts:
        # No need for to_blob or checking part type
        if part.text:
            all_responses.append(part.text)
            print(f'Bot: {" ".join(all_responses)}')
    os.system('pkill -f maincode.py')


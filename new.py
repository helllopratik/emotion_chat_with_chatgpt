import cv2
from keras.models import load_model
import numpy as np
import speech_recognition as sr
import threading
import time

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load the model
model = load_model('./facialemotionmodel.h5')
def get_camera_index():
    # Check available cameras to find a valid index
    for i in range(10):  # Try up to 10 cameras
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            cap.release()
            return i
    return None

# Function to detect emotion
def detect_emotion(image):
    # Emotion detection logic here...
    # Replace this with actual emotion detection based on image
    return 'Emotion'

# Function to interact with ChatGPT
def chat_with_gpt(text, emotion):
    print(f"Emotion: {emotion}")
    print(f"Query: {text}")
    # Use the text and detected emotion for further processing
    # Your ChatGPT interaction code here

def process_audio():
    while True:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            
            if "alexa" in query.lower():  # Check for the keyword "Alexa"
                print("Keyword 'Alexa' detected. Activating camera for emotion detection...")
                camera_index = get_camera_index()
                if camera_index is None:
                    print("No camera available.")
                    continue

                webcam = cv2.VideoCapture(camera_index)
                labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
                haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                face_cascade = cv2.CascadeClassifier(haar_file)

                while True:
                    ret, im = webcam.read()
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(im, 1.3, 5)

                    try:
                        for (p, q, r, s) in faces:
                            image = gray[q:q + s, p:p + r]
                            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
                            image = cv2.resize(image, (48, 48))
                            img = extract_features(image)
                            pred = model.predict(img)
                            prediction_label = labels[pred.argmax()]
                            print("Predicted Output:", prediction_label)
                            cv2.putText(im, '% s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

                        cv2.imshow("Output", im)
                        key = cv2.waitKey(1)

                        if key == ord('q'):
                            break

                    except cv2.error:
                        pass

                    # Stop camera and get the detected emotion
                    webcam.release()
                    detected_emotion = prediction_label
                    chat_with_gpt(query, detected_emotion)
                    break

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Error: {0}".format(e))

# Start the audio processing thread
audio_thread = threading.Thread(target=process_audio)
audio_thread.start()

import cv2
from keras.models import model_from_json
import numpy as np
import speech_recognition as sr
import threading
import time

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Function to detect emotion
def detect_emotion(image):
    # Emotion detection logic here...
    # Replace this with actual emotion detection based on image
    return 'Emotion'

# Function to interact with ChatGPT
def chat_with_gpt(text, emotion):
    # Interaction with ChatGPT logic here...
    # Use the text and detected emotion to communicate with ChatGPT
    print(f"Emotion: {emotion}")
    print(f"Query: {text}")
    # Replace this with your ChatGPT interaction code

# Load the facial emotion detection model and cascade classifier...
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Function to process video
def process_video():
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
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

# Function to process audio
def process_audio():
    while True:
        with sr.Microphone() as source:
            print("Say something:")
            audio = recognizer.listen(source)
        
        try:
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            
            # Capture frames for 2 seconds
            frames = []
            start_time = time.time()
            while (time.time() - start_time) < 2:
                _, frame = webcam.read()
                frames.append(frame)
            
            # Process the collected frames
            if frames:
                avg_frame = np.mean(frames, axis=0, dtype=np.uint8)
                emotion = detect_emotion(avg_frame)
                chat_with_gpt(query, emotion)
        
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Error: {0}".format(e))

# Run video processing in the main thread
video_thread = threading.Thread(target=process_video)
video_thread.start()

# Run audio processing in a separate thread
audio_thread = threading.Thread(target=process_audio)
audio_thread.start()

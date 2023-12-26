import cv2
from keras.models import model_from_json
import numpy as np
import speech_recognition as sr

# Load the facial emotion detection model and cascade classifier...
# Define functions for facial emotion detection...

# Initialize speech recognizer
recognizer = sr.Recognizer()

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
        
        # Speech recognition
        with sr.Microphone() as source:
            print("Say something:")
            audio = recognizer.listen(source)
            
        try:
            # Recognize speech using Google Speech Recognition
            query = recognizer.recognize_google(audio)
            print("You said:", query)
            
            # Here you can send the 'query' to ChatGPT or perform any other actions
            
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Error: {0}".format(e))
        
        if key == ord('q'):
            break
        
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()


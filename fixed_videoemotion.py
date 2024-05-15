import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Loading model and necessary setups (as provided in the original code)

def extract_features(image):
    # Function definition (as provided in the original code)
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion():
    # Capture video and detect emotion (as provided in the original code)
    # Replace the return statement with sending the detected emotion back to caller
    webcam = cv2.VideoCapture(0)
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    
    emotion_detected = False  # Flag to check if emotion is detected
    
    while not emotion_detected:
        _, im = webcam.read()
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
                
                # Set the flag to indicate emotion detection
                emotion_detected = True
                
            cv2.imshow("Output", im)
            cv2.waitKey(1)
            
        except cv2.error:
            pass

    return prediction_label

# Run emotion detection and return the detected emotion
detected_emotion = detect_emotion()

# Send the detected emotion to the caller (voicedetection.py)
print(detected_emotion)

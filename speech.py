import cv2
import speech_recognition as sr
import my_openai
from keras.models import model_from_json
import numpy as np

# Add your OpenAI API key here
my_openai.api_key = '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'

# Load the facial emotion detection model
# ... (Your model loading code here)

# Function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Open a webcam capture
webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Read video frames
    i, im = webcam.read()
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
        
        # Use SpeechRecognition library to recognize speech
        with sr.Microphone() as source:
            print("Say something!")
            audio = recognizer.listen(source)
        
        # Convert speech to text
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        
        # Combine the detected emotion and user query and send to ChatGPT
        chat_input = f"I am feeling {prediction_label}. {query}"
        response = my_openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "system", "content": chat_input}])
        chat_reply = response['choices'][0]['message']['content']
        
        print("ChatGPT reply:", chat_reply)
        
        cv2.waitKey(27)
    except cv2.error:
        pass

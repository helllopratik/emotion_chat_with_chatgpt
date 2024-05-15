import cv2
import numpy as np
import tensorflow as tf

# Load the emotion detection model (replace with your trained model)
model = tf.keras.models.load_model("facialemotionmodel.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (48, 48))
    img = extract_features(image)

    # Make predictions using your emotion detection model
    predictions = model.predict(img)
    emotion_index = np.argmax(predictions)  # Get the index of the highest probability
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

def process_video():
    webcam = cv2.VideoCapture(0)

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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()


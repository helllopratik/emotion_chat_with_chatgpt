import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("emotiondetector_mobilenetv2_v2.keras")

# Preprocess input image
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Real-time emotion detection
def detect_emotion(image):
    # Preprocess image
    preprocessed_image = preprocess_image(image)
    
    # Perform inference
    predictions = model.predict(preprocessed_image)
    
    # Get predicted emotion label
    predicted_label = emotion_labels[np.argmax(predictions)]
    
    return predicted_label

# Real-time video processing
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform emotion detection
    predicted_emotion = detect_emotion(frame)
    
    # Display predicted emotion label on the frame
    cv2.putText(frame, predicted_emotion, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Real-time Emotion Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

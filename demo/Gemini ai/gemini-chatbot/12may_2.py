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

chat = model.start_chat(history=[])  # Start a chat session

# Load the Haar cascade for face detection (not needed without GUI)
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)  # Commented out

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def extract_features(image):
    """Prepares an image for the emotion detection model."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for the model's input format
    return feature / 255.0  # Normalize pixel values


def detect_emotion(face_image):
 # Load the emotion detection model (replace with your trained model)
    model_emotion = tf.keras.models.load_model("emotiondetector.keras")
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)
  # Placeholder for the actual model
    if model_emotion is None:
        return None

    # Preprocess the image (resize, normalize, etc.)
    face_image = cv2.resize(face_image, (48, 48))
    img = extract_features(face_image)

    # Make predictions using your emotion detection model
    predictions = model_emotion.predict(img)
    emotion_index = np.argmax(predictions)  # Get the index of the highest probability
    detected_emotion = emotion_labels[emotion_index]

    return detected_emotion



def generate_paragraph_response(question, emotion):
    """Generates a paragraph response tailored to the user's question and detected emotion.

    Args:
        question: The user's question.
        emotion: The detected emotion label.

    Returns:
        A paragraph response from the chatbot that acknowledges the emotion.
    """

    emotion_to_response_style = {
        "angry": "I understand you're feeling angry. Is there anything I can do to help you calm down?",
        "disgust": "It sounds like something unpleasant is bothering you. Would you like to talk about it?",
        "fear": "Don't worry, I'm here for you. What are you afraid of?",
        "happy": "That's great to hear! What makes you happy today?",
        "neutral": "Tell me more about what's on your mind.",
        "sad": "I'm sorry to hear you're feeling sad. Is there anything I can do to help?",
        "surprise": "Wow, that's surprising! Tell me more about it.",
    }

    return emotion_to_response_style.get(emotion, "I'm still learning emotions. How can I help you?")


def main():
    while True:
        question = input("You: ")
        detected_emotion = detect_emotion(None)  # Placeholder for actual emotion detection
        response_paragraph = generate_paragraph_response(question, detected_emotion)
        print(f"Bot: {response_paragraph}")


if __name__ == "__main__":
    main()


import os
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
import numpy as np
import tkinter as tk
import tensorflow as tf
# Load environment variables (assuming you have a GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the generative AI model using the API key
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')  # You can experiment with other models

chat = model.start_chat(history=[])

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

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

def main():
    root = tk.Tk()
    root.title("Emotion Based Chat")

    question_label = tk.Label(root, text="You:")
    question_label.grid(row=0, column=0, padx=5, pady=5)

    question_entry = tk.Entry(root, width=50)
    question_entry.grid(row=0, column=1, padx=5, pady=5)

    detected_emotion_label = tk.Label(root, text="Detected Emotion:")
    detected_emotion_label.grid(row=1, column=0, padx=5, pady=5)

    detected_emotion_text = tk.StringVar()
    detected_emotion_display = tk.Label(root, textvariable=detected_emotion_text)
    detected_emotion_display.grid(row=1, column=1, padx=5, pady=5)

    chat_label = tk.Label(root, text="Bot:")
    chat_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def get_response(question, detected_emotion):
        response = chat.send_message(f"Consider my emotion as {detected_emotion}, {question}")  # Get the response from the chat

        # Extract and print the text from the response
        all_responses = []
        for part in response.parts:
            # No need for to_blob or checking part type
            if part.text:
                all_responses.append(part.text)
                chat_label.config(text=f'Bot: {" ".join(all_responses)}')

    def process_video():
        webcam = cv2.VideoCapture(0)
        detected_emotion = None

        while True:
            ret, frame = webcam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_image = gray[y:y + h, x:x + w]
                detected_emotion = detect_emotion(face_image)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Emotion: {detected_emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Emotion Detection", frame)
            if detected_emotion:  # If emotion is detected, break the loop
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()
        return detected_emotion

    def on_enter(event=None):
        question = question_entry.get()
        detected_emotion = None
        while detected_emotion not in emotion_labels:
            detected_emotion = process_video()
        detected_emotion_text.set(detected_emotion)
        get_response(question, detected_emotion)

    question_entry.bind("<Return>", on_enter)

    root.mainloop()

if __name__ == "__main__":
    main()


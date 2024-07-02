import os
from dotenv import load_dotenv
import google.generativeai as genai
import cv2
import numpy as np
import tkinter as tk
import ttkbootstrap as ttkb
import tensorflow as tf

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

question_entry = None  # Define question_entry as a global variable

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(face_image):
    model_emotion = tf.keras.models.load_model("emotiondetector.keras")
    if model_emotion is None:
        return None
    face_image = cv2.resize(face_image, (48, 48))
    img = extract_features(face_image)
    predictions = model_emotion.predict(img)
    emotion_index = np.argmax(predictions)
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

def start_chat(chat):
    global question_entry  # Declare question_entry as global
    root = tk.Tk()
    root.title("Emotion Based Chat")

    # Set the style/theme explicitly
    style = ttkb.Style()
    style.theme_use('darkly')  # or any other theme you prefer

    # Function to disable text entry and send button after sending the first message
    def disable_entry_and_button():
        question_entry.config(state="disabled")
        send_button.config(state="disabled")

    # Function to enable text entry and send button for asking again
    def enable_entry_and_button():
        question_entry.config(state="normal")
        send_button.config(state="normal")

    question_label = ttkb.Label(root, text="You:")
    question_label.grid(row=0, column=0, padx=5, pady=5)

    question_entry = ttkb.Entry(root, width=50)
    question_entry.grid(row=0, column=1, padx=5, pady=5)

    detected_emotion_label = ttkb.Label(root, text="Detected Emotion:")
    detected_emotion_label.grid(row=1, column=0, padx=5, pady=5)

    detected_emotion_text = tk.StringVar()
    detected_emotion_display = ttkb.Label(root, textvariable=detected_emotion_text)
    detected_emotion_display.grid(row=1, column=1, padx=5, pady=5)

    chat_label = ttkb.Label(root, text="Bot:")
    chat_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def get_response(question, detected_emotion):
        if detected_emotion == 'neutral':
            detected_emotion = 'peaceful'  # Treat neutral as peaceful

        response = chat.send_message(f"Consider my emotion as {detected_emotion}, {question}. response based on emotion.")

        all_responses = []
        for part in response.parts:
            if part.text:
                all_responses.append(part.text)
                chat_label.config(text=f'Bot: {" ".join(all_responses)}')

        disable_entry_and_button()
        ask_again_button.grid(row=3, column=1, padx=5, pady=5)

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
            if detected_emotion:
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

    def ask_again():
        root.destroy()
        global chat  # Declare chat as global to reassign it
        chat = model.start_chat(history=[])  # Start a new chat session
        start_chat(chat)  # Restart the chat

    send_button = ttkb.Button(root, text="Send", command=on_enter)
    send_button.grid(row=3, column=1, padx=5, pady=5)

    ask_again_button = ttkb.Button(root, text="Ask Again", command=ask_again)

    root.mainloop()

if __name__ == "__main__":
    chat = model.start_chat(history=[])
    start_chat(chat)


import os
import cv2
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import google.generativeai as genai
import subprocess
import re
import warnings
import tkinter as tk
from tkinter import scrolledtext, messagebox
from threading import Thread
import sys  # Import sys to handle restarting the program

warnings.filterwarnings("ignore")

# Load environment variables (assuming you have a GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the generative AI model using the API key
genai.configure(api_key=API_KEY)

# Initialize generative model for chat
model = genai.GenerativeModel('gemini-pro')

# Load the emotion detection model
model_emotion = tf.keras.models.load_model("emotiondetector.keras")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion(image):
    # Preprocess the image (resize, normalize, etc.)
    image = cv2.resize(image, (48, 48))
    img = extract_features(image)

    # Make predictions using the emotion detection model
    predictions = model_emotion.predict(img)
    emotion_index = np.argmax(predictions)  # Get the index of the highest probability
    detected_emotion = emotion_labels[emotion_index]
    return detected_emotion

def process_video():
    # Start video capture
    webcam = cv2.VideoCapture(0)
    detected_emotion = None

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
        if detected_emotion:  # If emotion is detected, break the loop
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    webcam.release()
    cv2.destroyAllWindows()
    return detected_emotion

# Function to execute a command and capture output
def execute_command(command):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return stdout.decode(), stderr.decode()
    except Exception as e:
        return '', str(e)

# Function to extract commands from response text
def extract_commands(response_text):
    # Regular expression to match commands related to file operations, system settings, and updates
    command_pattern = r'\b(echo\s+".*?"\s*>\s*\S+|brightnessctl\s+\S+|amixer\s+\S+|sudo\s+apt\s+update|sudo\s+apt\s+upgrade)'
    commands = re.findall(command_pattern, response_text, re.IGNORECASE)
    return commands

# Function to detect and save code snippet into appropriate file
def save_code_to_file(code_snippet):
    # Detect language from code snippet
    if "import" in code_snippet:
        # Assume it's Python code
        filename = "new.py"
    elif "echo" in code_snippet:
        # Assume it's a shell script
        filename = "new.sh"
    else:
        # Default to a generic filename
        filename = "new.txt"

    # Save code snippet to file
    with open(filename, 'w') as f:
        f.write(code_snippet)

    return filename

# GUI functions
def on_send():
    user_input = entry.get()
    if user_input:
        chat_area.config(state='normal')
        chat_area.insert(tk.END, f"You: {user_input}\n")
        chat_area.config(state='disabled')
        entry.delete(0, tk.END)
        
        # Disable the send button and start a thread to process the video and get the emotion
        send_button.config(state=tk.DISABLED)
        ask_again_button.config(state=tk.NORMAL)

        def detect_and_respond():
            detected_emotion = process_video()
            chat_area.config(state='normal')
            chat_area.insert(tk.END, f"Detected Emotion: {detected_emotion}\n")

            response = model.start_chat(history=[]).send_message(f"Consider my emotion as {detected_emotion}, {user_input}, (reply as second person Human-like conversation) in short")
            
            all_responses = []
            for part in response.parts:
                if part.text:
                    all_responses.append(part.text)
                    response_text = " ".join(all_responses)
                    chat_area.insert(tk.END, f'Bot: {response_text}\n')

                    commands = extract_commands(response_text)
                    for command in commands:
                        stdout, stderr = execute_command(command.strip())
                        if stdout:
                            chat_area.insert(tk.END, f"Command Output: {stdout}\n")
                            saved_file = save_code_to_file(stdout)
                            chat_area.insert(tk.END, f"Code saved to file: {saved_file}\n")
                            
                            if saved_file.endswith('.py'):
                                try:
                                    exec(open(saved_file).read())
                                    chat_area.insert(tk.END, f"Executed {saved_file} successfully.\n")
                                except Exception as e:
                                    chat_area.insert(tk.END, f"Error executing {saved_file}: {str(e)}\n")

                        if stderr:
                            chat_area.insert(tk.END, f"Command Error: {stderr}\n")
            chat_area.config(state='disabled')
        
        thread = Thread(target=detect_and_respond)
        thread.start()

def on_ask_again():
    root.destroy()
    # Restart the program
    os.execl(sys.executable, sys.executable, *sys.argv)

def on_close():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

# Create the main window
root = tk.Tk()
root.title("Emotion-Based Conversational Bot")
root.geometry("800x600")  # Set the initial size larger

# Create a frame for the chat area
frame = tk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a scrollable text area for the chat
chat_area = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state='disabled', height=25, width=100)
chat_area.pack(fill=tk.BOTH, expand=True)

# Create an entry box for user input
entry = tk.Entry(root, width=70)
entry.pack(padx=10, pady=10, fill=tk.X, expand=True)

# Create a send button
send_button = tk.Button(root, text="Send", command=on_send)
send_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create an ask again button
ask_again_button = tk.Button(root, text="Ask Again", command=on_ask_again, state=tk.DISABLED)
ask_again_button.pack(side=tk.LEFT, padx=10, pady=10)

# Set the on close event
root.protocol("WM_DELETE_WINDOW", on_close)

# Allow resizing
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)
frame.columnconfigure(0, weight=1)

# Start the GUI event loop
root.mainloop()


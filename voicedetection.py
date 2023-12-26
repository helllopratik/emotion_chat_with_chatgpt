import speech_recognition as sr
import subprocess

# Initialize speech recognizer
recognizer = sr.Recognizer()

def detect_keyword(voice_text):
    if "alexa" in voice_text.lower():
        return True
    return False

def run_video_emotion():
    emotion = subprocess.run(["python", "videoemotion.py"], capture_output=True, text=True)
    return emotion.stdout.strip()

while True:
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)
    
    try:
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        
        if detect_keyword(query):
            print("Keyword 'alexa' detected. Running emotion detection...")
            detected_emotion = run_video_emotion()
            print(f"Detected Emotion: {detected_emotion}\nQuery: {query}")
    
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))

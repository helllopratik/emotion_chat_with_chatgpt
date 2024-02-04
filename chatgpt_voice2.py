import os
import openai
import time
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import numpy as np
import speech_recognition as sr
import subprocess

openai.api_key = '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'
language = 'en'

r = sr.Recognizer()

name = "Pratik"
greetings = [f"whats up master {name}",
             "yeah?",
             "Well, hello there, Master of Puns and Jokes - how's it going today?",
             f"Ahoy there, Captain {name}! How's the ship sailing?",
             f"Bonjour, Monsieur {name}! Comment ça va? Wait, why the hell am I speaking French?"]

engine = pyttsx3.init("dummy")
voice = engine.getProperty('voices')[2]
engine.setProperty('voice', voice.id)

# Set the rate of speech (adjust this value as needed)
engine.setProperty('rate', 1500)  # Adjust the rate (words per minute)

def listen_for_wake_word(source):
    print("Listening for 'Hey'...")

    while True:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            if "hey" in text.lower():
                print("Wake word detected.")
                engine.say(np.random.choice(greetings))
                engine.runAndWait()
                listen_and_respond(source)
                break
        except sr.UnknownValueError:
            pass

def play_audio(filename):
    sound = AudioSegment.from_mp3(filename)
    play(sound)

def run_video_emotion():
    emotion = subprocess.run(["python", "videoemotion.py"], capture_output=True, text=True)
    return emotion.stdout.strip()

def listen_and_respond(source):
    print("Listening...")
    while True:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            detected_emotion = run_video_emotion()
            print(f"Detected Emotion: {detected_emotion}")
            emotions_mapping = {
            'happy': 'consider my emotion as happy',
            'sad': 'consider my emotion as sad',
            'angry': 'consider my emotion as angry',
            'surprise': 'consider my emotion as surprised',
            'fear': 'consider my emotion as feared',
            'neutral': 'consider my emotion as neutral'
            
        }

            emotion_context = emotions_mapping.get(detected_emotion, 'consider my emotion as neutral')
            a = f"{emotion_context} {text}"           
            # print(f"{detected_emotion}You said: {text} ")
            if not text:
                continue
            print(a)
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{a}"}])
            response_text = response.choices[0].message.content
            print(response_text)

            myobj = gTTS(text=response_text, lang=language, slow=False)
            myobj.save("response.mp3")
            
            play_audio("response.mp3")

            if not audio:
                listen_for_wake_word(source)
        except sr.UnknownValueError:
            time.sleep(2)
            print("Silence found, shutting up, listening...")
            listen_for_wake_word(source)
            break
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            listen_for_wake_word(source)
            break

with sr.Microphone() as source:
    listen_for_wake_word(source)

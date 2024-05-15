import os
from openai import OpenAI

client = OpenAI(api_key='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')
import time
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pyttsx3
import numpy as np
import subprocess
import requests

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

def listen_and_respond(source):
    print("Listening...")
    while True:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            if not text:
                continue

            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{text}"}])
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

def send_query_to_gpt(query):
    endpoint = "https://api.openai.com/v1/engines/davinci/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"  # Replace YOUR_API_KEY with your actual API key
    }

    data = {
        "prompt": query,
        "max_tokens": 50  # Adjust this based on the response length you want
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().choices[0].text
    else:
        return "Failed to get a response from ChatGPT"

def detect_keyword(voice_text):
    if "what" in voice_text.lower():
        return True
    return False

def run_video_emotion():
    emotion = subprocess.run(["python", "videoemotion.py"], capture_output=True, text=True)
    return emotion.stdout.strip()

def handle_microphone():
    with sr.Microphone() as source:
        print("Say something:")
        audio = r.listen(source)
    return audio

def run_combined():
    while True:
        try:
            audio = handle_microphone()
            query = r.recognize_google(audio)
            print("You said:", query)

            if detect_keyword(query):
                print("Keyword 'what' detected. Running emotion detection...")
                detected_emotion = run_video_emotion()
                print(f"Detected Emotion: {detected_emotion}\nQuery: {query}")
                a = 'asking in ' + detected_emotion + query
                response = send_query_to_gpt(a)
                print("ChatGPT:", response)
        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Error: {0}".format(e))

if __name__ == "__main__":
    with sr.Microphone() as source:
        listen_for_wake_word(source)

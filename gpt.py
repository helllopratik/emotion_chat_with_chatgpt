import speech_recognition as sr
from gtts import gTTS
import os
import requests

# Initialize speech recognizer
recognizer = sr.Recognizer()

def send_query_to_gpt(prompt):
    endpoint = "https://api.openai.com/v1/engines/davinci/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"  # Replace with your API key
    }

    data = {
        "prompt": prompt,
        "max_tokens": 150  # Adjust based on desired response length
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return "Failed to get a response from ChatGPT"

# Function to convert text to speech
def text_to_speech(text, speed=1.5):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")

# Function to play the speech audio
def play_audio():
    os.system("mpg321 response.mp3")

while True:
    try:
        with sr.Microphone() as source:
            print("Say something:")
            audio = recognizer.listen(source)
        
        query = recognizer.recognize_google(audio)
        print("You said:", query)

        prompt = query  # Use the user's query as the prompt for ChatGPT
        response = send_query_to_gpt(prompt)
        print("ChatGPT:", response)

        text_to_speech(response)
        play_audio()

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))

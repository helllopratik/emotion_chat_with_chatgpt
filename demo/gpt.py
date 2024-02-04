# Import necessary libraries
import speech_recognition as sr
import requests
import json
import pyttsx3

# Initialize the speech recognition engine
r = sr.Recognizer()

# Function to convert voice into text
def convert_voice_to_text():
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand.")
        except sr.RequestError as e:
            print("Sorry, I could not request results from Google Speech Recognition service; {0}".format(e))

# Function to send text to ChatGPT and get response
def send_text_to_chatgpt(text):
    url = "https://api.openai.com/v1/engines/davinci-codex/completions"
    payload = {
        "prompt": text,
        "max_tokens": 50
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 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"
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    return response.json()["choices"][0]["text"]

# Function to play the response as audio
def play_audio(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Main program
if __name__ == "__main__":
    # Convert voice into text
    voice_text = convert_voice_to_text()
    
    # Send text to ChatGPT and get response
    chatgpt_response = send_text_to_chatgpt(voice_text)
    
    # Play the response as audio
    play_audio(chatgpt_response)


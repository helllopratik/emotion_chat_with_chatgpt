import os
import openai
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

# OpenAI API key
openai.api_key = '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'

# Function to convert text to speech
def text_to_speech(text, filename="response.mp3", speed=1.5):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

# Function to convert speech to text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Error: {e}"

# Function to interact with ChatGPT
def chat_with_gpt(query):
    response = openai.Completion.create(
        engine="davinci",
        prompt=query,
        max_tokens=50
    )
    return response.choices[0].text.strip()

# Record audio and save it as an MP3 file
def record_audio(file_name):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording...")
        audio = recognizer.listen(source)
    with open(file_name, "wb") as f:
        f.write(audio.get_wav_data())

# Entry point
if __name__ == "__main__":
    # Record audio and save as mp3 file
    audio_file_name = "audio_query.mp3"
    record_audio(audio_file_name)

    # Convert audio to text
    text = speech_to_text(audio_file_name)
    print("You said:", text)

    # Interact with ChatGPT
    chat_response = chat_with_gpt(text)

    # Convert ChatGPT response to speech and play it
    text_to_speech(chat_response, "response.mp3")
    playsound("response.mp3")

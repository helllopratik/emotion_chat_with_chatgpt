import speech_recognition as sr
import subprocess
import requests

# Initialize speech recognizer
recognizer = sr.Recognizer()

def send_query_to_gpt(query):
    endpoint = "https://api.openai.com/v1/engines/davinci/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-wWKT##########################00zsV4KG7"  # Replace YOUR_API_KEY with your actual API key
    }

    data = {
        "prompt": query,
        "max_tokens": 50  # Adjust this based on the response length you want
    }

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return "Failed to get a response from ChatGPT"

def detect_keyword(voice_text):
    if "what" in voice_text.lower():
        return True
    return False

def run_video_emotion():
    emotion = subprocess.run(["python", "videoemotion.py"], capture_output=True, text=True)
    return emotion.stdout.strip()


# Function to handle microphone start and stop
def handle_microphone():
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)
    return audio


while True:
    try:
        audio = handle_microphone()
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        
        if detect_keyword(query):
            print("Keyword 'alexa' detected. Running emotion detection...")
            detected_emotion = run_video_emotion()
            print(f"Detected Emotion: {detected_emotion}\nQuery: {query}")
            a = detected_emotion + query
            response = send_query_to_gpt(query)
            print("ChatGPT:", response)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error: {0}".format(e))
    # with sr.Microphone() as source:
    #     print("Say something:")
    #     audio = recognizer.listen(source)
    
    # try:
    #     query = recognizer.recognize_google(audio)
    #     print("You said:", query)
        
    #     if detect_keyword(query):
    #         print("Keyword 'alexa' detected. Running emotion detection...")
    #         detected_emotion = run_video_emotion()
    #         print(f"Detected Emotion: {detected_emotion}\nQuery: {query}")
    #         a= detected_emotion + query
    #         response = send_query_to_gpt(a)
    #         print("ChatGPT:", response)
    # except sr.UnknownValueError:
    #     print("Could not understand audio")
    # except sr.RequestError as e:
    #     print("Error: {0}".format(e))

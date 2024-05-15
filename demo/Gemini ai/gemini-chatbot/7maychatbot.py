# chatbot.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
from subprocess import check_output

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

while(True):
    # Detect emotion
    detected_emotion = check_output(["python", "7mayreal1.py"]).decode("utf-8").strip()

    # Append detected emotion to the instruction
    instruction = f"In this chat, respond based on the emotion passed by the user. Consider my emotion as {detected_emotion}. "
    question = input("You: ")
    response = chat.send_message(instruction + question)
    print('\n')
    print(f"Bot: {response.text}")
    print('\n')


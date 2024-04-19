import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')


genai.configure(api_key=API_KEY
)

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])
instruction = "In this chat, respond based on the emotion passed by the user. "
while(True):
  question = input("You: ")
  response = chat.send_message(instruction + question)
  print('\n')
  print(f"Bot: {response.text}")
  print('\n')
  instruction = ''
  

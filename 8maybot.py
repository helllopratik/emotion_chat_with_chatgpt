import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (assuming you have a GEMINI_API_KEY)
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

# Configure the generative AI model using the API key
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel('gemini-pro')  # You can experiment with other models

chat = model.start_chat(history=[])

while True:
  question = input("You: ")
  response = chat.send_message(question)
  print('\n')
  print(f"Bot: {response.text}")
  print('\n')


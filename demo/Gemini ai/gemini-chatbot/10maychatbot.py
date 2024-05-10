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
  response = chat.send_message(question)  # Get the response from the chat

  # Extract and print the text from the response
  all_responses = []
  for part in response.parts:
    # No need for to_blob or checking part type
    if part.text:
       all_responses.append(part.text)
       print(f'Bot: {" ".join(all_responses)}')
  os.system('pkill -f 10maychatbot.py')

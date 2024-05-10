import os
print("Human like conversation based on Emotion of User")

def run_gemini():
  """Prompts the user for input and sends it to Gemini."""

  while True:
   
    #question = input("You: ")
    # Your logic to send the question to Gemini and process the response
    os.system('python 10maychatbot.py')
   
    # Terminate the script (optional)
    #os.system('python 10maychatbot.py')  # Use with caution, consider #exit()
    #break

if __name__ == "__main__":

  run_gemini()


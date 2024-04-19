import requests

# Replace with your Generative Language Client API key
API_KEY = "AIzaSyCz--rev575b2SsmM_eoSPrUpHRRtd_slg"

def send_query(query):
  """
  Sends a query to Gemini AI and returns the response.

  Args:
      query: The user's query string.

  Returns:
      The Gemini AI response as a string, or None on error.
  """
  url = "https://api.gemini.ai/v1/ask"
  headers = {"Authorization": f"Bearer {API_KEY}"}
  payload = {"prompt": query, "role": "USER"}

  try:
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()  # Raise an exception for non-200 status codes
    return response.json()["text"]
  except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
    return None

# Welcome message
print("Welcome to Gemini AI Chat!")

while True:
  # Get user query
  user_query = input("You: ")

  # Send query to Gemini AI
  response = send_query(user_query)

  if response:
    print(f"Gemini AI: {response}")

    # Additional functionalities based on the video
    # (e.g., user response processing, handling different commands)

  else:
    print("Failed to get response from Gemini AI.")

print("Goodbye!")


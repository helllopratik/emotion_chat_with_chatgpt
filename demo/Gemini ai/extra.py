import requests

# Replace with your Generative Language Client API key
API_KEY = "AIzaSyCz--rev575b2SsmM_eoSPrUpHRRtd_slg"

# Function to send query to Gemini AI with error handling
def send_query(query):
  """
  Sends a query to Gemini AI and returns the response. Handles potential errors.

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

# Get user input
user_query = input("Enter your question: ")

# Send query to Gemini AI and display response
response = send_query(user_query)

if response:
  print(f"Gemini AI response: {response}")
else:
  print("Failed to get response from Gemini AI.")


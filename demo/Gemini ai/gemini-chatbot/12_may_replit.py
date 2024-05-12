import requests
import json

# Define API endpoint and API key
api_endpoint = "https://api.gemini.ai/v1/chat"
api_key = "API"

# Function to send query and get response
def send_query(query):
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    data = {"query": query}
    response = requests.post(api_endpoint, headers=headers, json=data)
    return response.json()

# Get user input
query = input("Ask your question: ")

# Send query to Gemini AI and get response
response = send_query(query)

# Print chatbot response
print(response["response"])

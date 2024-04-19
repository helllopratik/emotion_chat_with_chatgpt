import requests
import json
import ssl  # Import the ssl module

# Replace with your actual Gemini API key
GEMINI_API_KEY = "AIzaSyDRBfR0Lhe2nNGMmWL6c7XV64cW3XP_yRc"

def get_gemini_response(prompt):
    try:
        response = requests.post(
            "https://api.gemini.ai/v1/ask",
            json={"prompt": prompt, "api_key": GEMINI_API_KEY},
            timeout=10,  # Set a reasonable timeout
        )
        response.raise_for_status()  # Raise an exception if the request fails
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return "An error occurred while fetching the response."

def main():
    print("Welcome to the Gemini Chatbot!")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        gemini_response = get_gemini_response(user_input)
        print(f"Chatbot: {gemini_response}")

if __name__ == "__main__":
    main()


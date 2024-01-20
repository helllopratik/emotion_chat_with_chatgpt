import subprocess
import openai
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

openai.api_key = 'Your-API-KEY'

def run_video_emotion():
    try:
        # Run the external script to detect emotion (modify this part as needed)
        # Replace this section with your text-based emotion recognition code
        # For example, using sentiment analysis from NLTK library
        text = "I'm feeling sad"
        emotion = detect_emotion(text)
        return emotion
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None

def detect_emotion(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:
        return "happy"
    elif sentiment_score['compound'] <= -0.05:
        return "sad"
    elif sentiment_score['compound'] <= -0.25:
        return "fear"
    elif sentiment_score['compound'] >= 0.25:
        return "surprise"
    else:
        return "neutral"

def get_emotion_context(detected_emotion):
    emotions_list = ['happy', 'sad', 'fear', 'neutral', 'surprise']
    detected_emotion = detected_emotion.lower()
    for emotion in emotions_list:
        if emotion in detected_emotion:
            return f"consider my emotion as {emotion}"
    return "consider my emotion as neutral"

def query_chat_gpt(query, emotion_context):
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{emotion_context} {query}"}])
        response_text = response.choices[0].message.content
        return response_text
    except Exception as e:
        print(f"Error querying ChatGPT: {e}")
        return None

def main():
    while True:
        try:
            user_query = input("Enter your query (or 'exit' to quit): ")
            if user_query.lower() in ['exit', 'quit']:
                break
            
            detected_emotion = run_video_emotion()
            emotion_context = get_emotion_context(detected_emotion)
            response = query_chat_gpt(user_query, emotion_context)
            if response:
                print(f"ChatGPT Response: {response}")
            else:
                print("Failed to get a response from ChatGPT")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

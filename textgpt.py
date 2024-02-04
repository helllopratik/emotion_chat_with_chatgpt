import subprocess
import openai

openai.api_key = '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'

def run_video_emotion():
    try:
        # Run the external script to detect emotion
        emotion = subprocess.run(["python", "videoemotion.py"], capture_output=True, text=True)
        detected_emotion = emotion.stdout.strip()
        return detected_emotion
    except Exception as e:
        print(f"Error running videoemotion.py: {e}")
        return None

def get_emotion_context(detected_emotion):
    # List of emotions to look for in the detected_emotion string
    emotions_list = ['happy', 'sad', 'fear', 'neutral', 'surprise']

    # Split the detected_emotion string and find the emotion in the list
    detected_emotion = detected_emotion.lower()
    for emotion in emotions_list:
        if emotion in detected_emotion:
            print(emotion)
            return f"consider my emotion as {emotion} because"

    # If no specific emotion is found, default to neutral
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
            #print({detected_emotion})
            response = query_chat_gpt(detected_emotion,user_query)
            if response:
                print(f"ChatGPT Response: {response}")
            else:
                print("Failed to get a response from ChatGPT")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

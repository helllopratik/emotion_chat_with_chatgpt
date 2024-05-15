import google.generativeai as genai

GOOGLE_API_KEY = 'API_KEY'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.0-pro-latest')
convo = model.start_chat()

while True:
    user_input = input('Gemini Prompt: ')
    convo.send_message(user_input)
    print(convo.last.text)
#response = model.generate_content(input('Ask Gemini: '))
#print(response)

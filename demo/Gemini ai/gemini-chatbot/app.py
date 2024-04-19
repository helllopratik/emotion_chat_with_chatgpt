import google.generativeai as genai

class GeniAIException(Exception):
"""GenAI Exception base class"""

class ChatBot:
""" Chat can only have one candidate count """
CHATBOT_NAME  = 'My Gemini AI'

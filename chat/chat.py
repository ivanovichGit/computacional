from openai import OpenAI
import os
from dotenv import load_dotenv
from ollama import chat
from ollama import ChatResponse

SYSTEM_MESSAGE = "You are a chatbot. You will have a conversation with a user. Be friendly and concise"

if __name__ == "__main__":
    load_dotenv()
    URL = os.environ.get('OPENAI_BASE_URL')
    KEY = os.environ.get('OPENAI_KEY')
    MODEL = os.environ.get('MODEL')

    client = OpenAI(
        base_url=URL,
        api_key=KEY,
    )

    print(f"Chatting with {MODEL} model at {URL}")

    # System instruction 
    messages=[
                {'role': 'system', 'content': SYSTEM_MESSAGE}
    ]
    
    while True:
        message = input("> ")

        # Append historial 
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model = MODEL, 
            messages = messages
        )

        assistant_text = response.choices[0].message.content
        print(assistant_text)

        # Para que también recuerde su respuesta 
        messages.append({'role': 'assistant', 'content': assistant_text})
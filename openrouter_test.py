import os
from dotenv import load_dotenv
from openrouter import OpenRouter


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenRouter(api_key=api_key)

messages = []

print("Sending message to OpenRouter...")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break
    
    messages.append({"role": "user", "content": user_input})

    response = client.chat.send(
        model="upstage/solar-pro-3:free",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        stream=False
    )

    reply_content = response.choices[0].message.content.strip()
    print(f"OpenRouter: {reply_content}\n")
    messages.append({"role": "assistant", "content": reply_content})
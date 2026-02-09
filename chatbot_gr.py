#chatbot_ui.py
import gradio as gr
from openrouter import OpenRouter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenRouter(api_key=api_key)

# Define the chat function (called by Gradio on each user message)
def chat_with_ai(message, history):
    # Convert Gradio history to OpenRouter format
    messages = []
    for msg in history:
        if isinstance(msg, (tuple, list)) and len(msg) >= 2:
            # Extract user and assistant messages from history
            user_msg = msg[0] if msg[0] else ""
            ai_msg = msg[1] if len(msg) > 1 and msg[1] else ""
            
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})

    # Add the new user message
    messages.append({"role": "user", "content": message})

    try:
        # Call OpenRouter
        response = client.chat.send(
            model="upstage/solar-pro-3:free",  # Use your preferred free model
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
        )
        
        ai_reply = response.choices[0].message.content.strip()
        return ai_reply

    except Exception as e:
        return f"Error: {str(e)}"

# Set up Gradio interface
with gr.Blocks(title="OpenRouter Chatbot") as demo:
    gr.Markdown("# AI Chatbot Powered by OpenRouter")
    gr.Markdown("Type your message below and chat with the AI!")
    
    chatbot = gr.ChatInterface(
        fn=chat_with_ai,
        examples=["What is the meaning of life?", "Tell me a joke."],
        title="Chat with AI",
        description="This chatbot uses OpenRouter's free models. Conversation history is maintained.",
        # theme=gr.themes.Soft()  # Nice visual theme (optional)
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)  # Se

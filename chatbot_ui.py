#chatbot_ui.py
import gradio as gr
from openrouter import OpenRouter
from dotenv import load_dotenv
import os
from pathlib import Path
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenRouter(api_key=api_key)

# Load local document
def load_document(doc_path="journal.pdf"):
    """Load document from local file (supports txt and pdf)"""
    if not Path(doc_path).exists():
        return "No document found. Please add document.pdf or document.txt"
    
    try:
        if doc_path.endswith('.pdf'):
            if PyPDF2 is None:
                return "PyPDF2 not installed. Run: pip install PyPDF2"
            
            text = ""
            with open(doc_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text if text else "Could not extract text from PDF"
        
        elif doc_path.endswith('.txt'):
            with open(doc_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        else:
            return "Unsupported file format. Use .txt or .pdf"
    
    except Exception as e:
        return f"Error loading document: {str(e)}"

# Store document content
document_content = load_document()

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

    # Create system prompt with document context
    system_prompt = f"""You are a helpful assistant. Answer questions based ONLY on the following document content. 
If the answer is not in the document, say "I don't have that information in the provided document."

Document Content:
---
{document_content}
---

Please answer the user's question based only on the above document."""

    # Add system message at the beginning
    messages.insert(0, {"role": "system", "content": system_prompt})
    
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
    gr.Markdown("This chatbot answers questions based on the loaded document only.")
    
    with gr.Row():
        # with gr.Column(scale=1):
        #     gr.Markdown("### Document Content")
        #     gr.Textbox(value=document_content[:500] + "..." if len(document_content) > 500 else document_content, 
        #               label="Loaded Document (first 500 chars)", lines=10, interactive=False)
        
        with gr.Column(scale=1):
            chatbot = gr.ChatInterface(
                fn=chat_with_ai,
                examples=["What is this document about?", "Summarize the key points."],
                title="Chat with AI",
                description="Ask questions about the document.",
            )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=True)  # Serve on all interfaces

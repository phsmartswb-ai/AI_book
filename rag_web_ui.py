"""
RAG Web UI using Gradio
Simple web interface for RAG application
"""

import gradio as gr
from simple_rag import SimpleRAG
from pathlib import Path

# Initialize RAG
rag = SimpleRAG()

def answer_question(question: str) -> str:
    """Answer a question using RAG"""
    if not question.strip():
        return "Please enter a question."
    
    return rag.answer_question(question, verbose=False)

def upload_document(file) -> str:
    """Handle document upload"""
    if file is None:
        return "No file uploaded"
    
    try:
        # Save uploaded file
        file_path = file.name
        rag._load_document(file_path)
        rag._chunk_documents()
        
        return f"✓ Document uploaded and processed: {Path(file_path).name}\nTotal chunks: {len(rag.chunks)}"
    except Exception as e:
        return f"Error uploading document: {str(e)}"

def get_stats() -> str:
    """Get RAG statistics"""
    stats = f"""
📊 RAG System Statistics
━━━━━━━━━━━━━━━━━━━━━━━
Documents loaded: {len(rag.documents)}
Text chunks: {len(rag.chunks)}
Documents:
"""
    for doc in rag.documents:
        stats += f"  • {doc['path']} ({doc['type'].upper()})\n"
    
    return stats

# Build Gradio interface
with gr.Blocks(title="RAG Chat Interface") as demo:
    gr.Markdown("# 🤖 RAG (Retrieval-Augmented Generation) Chat")
    gr.Markdown("Ask questions about your documents. The AI will search and cite relevant information.")
    
    with gr.Row():
        # with gr.Column():
        #     gr.Markdown("## 📚 Document Management")
            
        #     stats_display = gr.Textbox(
        #         label="System Stats",
        #         value=get_stats(),
        #         interactive=False,
        #         lines=8
        #     )
            
        #     upload_file = gr.File(label="Upload Document (.txt or .pdf)")
        #     upload_btn = gr.Button("Process Document")
        #     upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
        #     upload_btn.click(upload_document, inputs=[upload_file], outputs=[upload_status])
        
        with gr.Column():
            gr.Markdown("## 💬 Ask Questions")
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the documents...",
                lines=3
            )
            
            submit_btn = gr.Button("Get Answer", variant="primary")
            
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=6
            )
            
            submit_btn.click(answer_question, inputs=[question_input], outputs=[answer_output])

    with gr.Row():
        with gr.Column():
            gr.Markdown("Suggested Questions:")

            suggested_questions = [
                "What is the MVP?",
                "What city is analyzed and why?",
                "What are the key projections?",
                "What makes this project unique?",
                "What is the expected timeline?"
            ]
            
            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=6
            )
            
            for sq in suggested_questions:
                btn = gr.Button(sq, variant="secondary")
                btn.click(answer_question, inputs=[gr.Textbox(value=sq, visible=False)], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch()

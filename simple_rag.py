"""
Simple RAG (Retrieval-Augmented Generation) Application
Retrieves relevant information from documents and uses AI to answer questions
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

# Load environment variables
load_dotenv()

class SimpleRAG:
    def __init__(self, doc_paths: List[str] = None):
        """
        Initialize RAG with document paths
        
        Args:
            doc_paths: List of file paths (.txt, .pdf) to load as knowledge base
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "openrouter/auto"  # Uses best available model
        self.documents = []
        self.chunks = []
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        
        # Default document paths
        if doc_paths is None:
            doc_paths = ["document1.pdf", "document2.pdf", "document3.pdf", "document4.pdf"] # Add your document paths here
        
        # Load documents
        for doc_path in doc_paths:
            if Path(doc_path).exists():
                self._load_document(doc_path)
        
        if not self.documents:
            print("⚠️  No documents loaded. Please provide document paths.")
        else:
            print(f"✓ Loaded {len(self.documents)} documents")
            self._chunk_documents()
            print(f"✓ Created {len(self.chunks)} chunks for retrieval")
    
    def _load_document(self, file_path: str):
        """Load a document (txt or pdf)"""
        try:
            if file_path.endswith('.pdf'):
                if PyPDF2 is None:
                    print(f"⚠️  PyPDF2 not installed, skipping {file_path}")
                    return
                
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                
                self.documents.append({
                    "path": file_path,
                    "type": "pdf",
                    "content": text
                })
                print(f"  ✓ Loaded PDF: {file_path}")
            
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.documents.append({
                    "path": file_path,
                    "type": "txt",
                    "content": content
                })
                print(f"  ✓ Loaded TXT: {file_path}")
        
        except Exception as e:
            print(f"  ✗ Error loading {file_path}: {str(e)}")
    
    def _chunk_documents(self, chunk_size: int = 500, overlap: int = 100):
        """
        Split documents into chunks for efficient retrieval
        
        Args:
            chunk_size: Number of characters per chunk
            overlap: Number of characters to overlap between chunks
        """
        self.chunks = []
        
        for doc in self.documents:
            content = doc["content"]
            
            # Split into sentences for better chunking
            sentences = content.replace("\n", " ").split(". ")
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        self.chunks.append({
                            "content": current_chunk.strip(),
                            "source": doc["path"],
                            "type": doc["type"]
                        })
                    current_chunk = sentence + ". "
            
            # Add remaining chunk
            if current_chunk.strip():
                self.chunks.append({
                    "content": current_chunk.strip(),
                    "source": doc["path"],
                    "type": doc["type"]
                })
    
    def _retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Simple retrieval: find most relevant chunks using keyword matching
        For production, consider using embeddings or semantic search
        
        Args:
            query: User question
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        query_words = set(query.lower().split())
        
        # Score chunks based on keyword overlap
        scored_chunks = []
        for chunk in self.chunks:
            content_words = set(chunk["content"].lower().split())
            overlap = len(query_words & content_words)
            
            if overlap > 0:
                scored_chunks.append({
                    **chunk,
                    "score": overlap
                })
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
    
    def answer_question(self, question: str, verbose: bool = False) -> str:
        """
        Answer a question using RAG
        
        Args:
            question: User question
            verbose: Print retrieval details
            
        Returns:
            AI-generated answer based on retrieved documents
        """
        if not self.chunks:
            return "❌ No documents loaded. Cannot answer question."
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self._retrieve_relevant_chunks(question)
        
        if not retrieved_chunks:
            return "❌ No relevant information found in documents."
        
        if verbose:
            print(f"\n📚 Retrieved {len(retrieved_chunks)} relevant chunks:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"  {i}. Source: {chunk['source']} (score: {chunk['score']})")
                print(f"     {chunk['content'][:100]}...")
        
        # Step 2: Build context from retrieved chunks
        context = "\n\n".join([f"[Source: {chunk['source']}]\n{chunk['content']}" 
                               for chunk in retrieved_chunks])
        
        # Step 3: Generate answer using AI
        prompt = f"""Based on the following information from documents, answer the question:

DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions based on provided documents. Cite your sources."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": 1000
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            answer = result["choices"][0]["message"]["content"]
            return answer
        
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def interactive_chat(self):
        """Start an interactive Q&A session"""
        print("\n" + "="*60)
        print("RAG Interactive Chat")
        print("="*60)
        print("Ask questions about the loaded documents.")
        print("Type 'exit' or 'quit' to end.\n")
        
        while True:
            question = input("Q: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n🤖 Thinking...\n")
            answer = self.answer_question(question, verbose=True)
            print(f"\nA: {answer}\n")
    
    def save_qa_pairs(self, qa_pairs: List[Dict[str, str]], filename: str = "rag_qa.json"):
        """Save question-answer pairs to file"""
        with open(filename, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"✓ Saved {len(qa_pairs)} Q&A pairs to {filename}")


def main():
    """Example usage"""
    print("="*60)
    print("Simple RAG Application")
    print("="*60)
    
    # Initialize RAG with default documents
    rag = SimpleRAG()
    
    if not rag.chunks:
        print("\n⚠️  No documents to work with.")
        print("Please add .txt or .pdf files to the current directory")
        return
    
    # Example questions
    example_questions = [
        "What is the main topic of the documents?",
        "Summarize the key points",
        "What information is available?"
    ]
    
    print("\n" + "="*60)
    print("Example Q&A")
    print("="*60)
    
    qa_results = []
    for question in example_questions:
        print(f"\n❓ Q: {question}")
        answer = rag.answer_question(question, verbose=False)
        print(f"✓ A: {answer}\n")
        qa_results.append({"question": question, "answer": answer})
    
    # Start interactive session
    print("\n" + "="*60)
    rag.interactive_chat()


if __name__ == "__main__":
    main()

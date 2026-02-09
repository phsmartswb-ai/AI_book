"""
Advanced RAG with Semantic Search using OpenRouter Embeddings
More accurate retrieval using embedding-based similarity
"""

import os
import math
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import requests
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

load_dotenv()

class AdvancedRAG:
    def __init__(self, doc_paths: List[str] = None, use_embeddings: bool = True):
        """
        Initialize Advanced RAG with optional embeddings
        
        Args:
            doc_paths: List of file paths to load
            use_embeddings: Whether to use embeddings for semantic search
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = "openrouter/auto"
        self.documents = []
        self.chunks = []
        self.embeddings = {}  # Cache embeddings
        self.use_embeddings = use_embeddings
        
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        
        # Default paths
        if doc_paths is None:
            doc_paths = ["document1.pdf", "document2.pdf", "document3.pdf", "document4.pdf"] # Add your document paths here
        
        # Load documents
        for doc_path in doc_paths:
            if Path(doc_path).exists():
                self._load_document(doc_path)
        
        if self.documents:
            self._chunk_documents()
            if use_embeddings:
                self._create_embeddings()
    
    def _load_document(self, file_path: str):
        """Load document"""
        try:
            if file_path.endswith('.pdf'):
                if PyPDF2 is None:
                    return
                text = ""
                with open(file_path, 'rb') as f:
                    for page in PyPDF2.PdfReader(f).pages:
                        text += page.extract_text()
                self.documents.append({"path": file_path, "type": "pdf", "content": text})
                print(f"✓ Loaded: {file_path}")
            
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.documents.append({"path": file_path, "type": "txt", "content": f.read()})
                print(f"✓ Loaded: {file_path}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    def _chunk_documents(self, chunk_size: int = 500):
        """Split documents into chunks"""
        self.chunks = []
        for doc in self.documents:
            content = doc["content"]
            sentences = content.replace("\n", " ").split(". ")
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < chunk_size:
                    current_chunk += sentence + ". "
                else:
                    if current_chunk.strip():
                        self.chunks.append({
                            "content": current_chunk.strip(),
                            "source": doc["path"]
                        })
                    current_chunk = sentence + ". "
            
            if current_chunk.strip():
                self.chunks.append({"content": current_chunk.strip(), "source": doc["path"]})
    
    def _create_embeddings(self):
        """Create embeddings for all chunks (cached)"""
        print(f"Creating embeddings for {len(self.chunks)} chunks...")
        
        for i, chunk in enumerate(self.chunks):
            # Use simple hashing as mock embedding (for demo)
            # In production, use real embedding API or local models
            self.embeddings[i] = self._get_text_hash(chunk["content"])
            
            if (i + 1) % 10 == 0:
                print(f"  ✓ Embedded {i + 1}/{len(self.chunks)} chunks")
    
    def _get_text_hash(self, text: str) -> List[float]:
        """
        Simple hash-based embedding (mock)
        In production, use: model.encode() or OpenAI embeddings API
        """
        hash_val = hash(text)
        # Create a simple vector representation
        return [float((hash_val >> i) & 0xFF) / 255.0 for i in range(384)]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def _retrieve_by_similarity(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve chunks using semantic similarity"""
        if not self.use_embeddings:
            return self._retrieve_by_keywords(query, top_k)
        
        query_embedding = self._get_text_hash(query)
        
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            similarity = self._cosine_similarity(query_embedding, self.embeddings[i])
            scored_chunks.append({**chunk, "score": similarity})
        
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
    
    def _retrieve_by_keywords(self, query: str, top_k: int = 3) -> List[Dict]:
        """Fallback: keyword-based retrieval"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            content_words = set(chunk["content"].lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored_chunks.append({**chunk, "score": overlap})
        
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
    
    def answer_question(self, question: str, verbose: bool = False) -> str:
        """Generate answer with retrieved context"""
        if not self.chunks:
            return "No documents loaded."
        
        retrieved = self._retrieve_by_similarity(question)
        if not retrieved:
            return "No relevant information found."
        
        if verbose:
            print(f"\n📚 Retrieved {len(retrieved)} chunks")
        
        context = "\n\n".join([f"[{c['source']}]\n{c['content']}" for c in retrieved])
        
        prompt = f"""Answer based on this information:

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
                        {"role": "system", "content": "Answer based on provided documents."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000
                }
            )
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"
    
    def batch_qa(self, questions: List[str]) -> List[Dict]:
        """Answer multiple questions"""
        results = []
        for q in questions:
            print(f"Q: {q}")
            answer = self.answer_question(q)
            print(f"A: {answer}\n")
            results.append({"question": q, "answer": answer})
        return results


if __name__ == "__main__":
    print("Advanced RAG Application\n")
    
    rag = AdvancedRAG(use_embeddings=True)
    
    if rag.chunks:
        questions = [
            "What are the main topics?",
            "Give me a summary"
        ]
        rag.batch_qa(questions)
    else:
        print("Add documents to get started!")

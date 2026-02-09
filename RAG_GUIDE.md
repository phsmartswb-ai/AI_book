# RAG (Retrieval-Augmented Generation) Applications

A collection of RAG implementations from simple to advanced, with web UI.

## 📋 What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that:
1. **Retrieves** relevant information from your documents
2. **Augments** the AI prompt with this information
3. **Generates** accurate answers based on actual document content

**Benefits:**
- ✓ Accurate answers based on your data
- ✓ Cites document sources
- ✓ Reduces AI hallucinations
- ✓ Works with any document type

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Documents
Place documents in the current directory:
- `.txt` files (plain text)
- `.pdf` files (requires PyPDF2)

Examples already available:
- `document.txt`
- `example.pdf`
- `journal.pdf`

### 3. Run RAG

**Simple RAG (Keyword-based):**
```bash
python simple_rag.py
```

**Advanced RAG (Semantic search):**
```bash
python advanced_rag.py
```

**Web UI:**
```bash
python rag_web_ui.py
```

## 📚 RAG Versions

### `simple_rag.py` - Basic RAG
**Best for:** Learning, quick prototyping, small datasets

Features:
- Keyword-based retrieval
- Simple chunking strategy
- Interactive Q&A session
- Easy to understand and modify

Usage:
```python
from simple_rag import SimpleRAG

rag = SimpleRAG(doc_paths=["document.txt", "example.pdf"])

# Ask a question
answer = rag.answer_question("What is the main topic?")
print(answer)

# Interactive chat
rag.interactive_chat()
```

### `advanced_rag.py` - Semantic RAG
**Best for:** Production use, better accuracy, larger datasets

Features:
- Embedding-based retrieval (mock embeddings)
- Cosine similarity matching
- Batch Q&A processing
- Better accuracy than keyword matching

Usage:
```python
from advanced_rag import AdvancedRAG

rag = AdvancedRAG(use_embeddings=True)

# Batch process questions
questions = [
    "What's the summary?",
    "Who is mentioned?",
    "What are the key findings?"
]

results = rag.batch_qa(questions)
```

### `rag_web_ui.py` - Web Interface
**Best for:** User-friendly access, document upload, team use

Features:
- Beautiful web interface with Gradio
- Document upload capability
- Real-time statistics
- Easy sharing and deployment

Usage:
```bash
python rag_web_ui.py
# Open browser to http://localhost:7860
```

## 🔄 RAG Architecture

```
User Question
    ↓
[Retrieval] → Search documents for relevant chunks
    ↓
[Ranking] → Score and sort by relevance
    ↓
[Context Augmentation] → Build prompt with top chunks
    ↓
[Generation] → Send to AI model with context
    ↓
Answer with Sources
```

## 💡 How It Works

### Step 1: Document Chunking
Documents are split into manageable pieces:
```
Full Document (10,000 words)
    ↓
Split into chunks (500 words each, with overlap)
    ↓
Chunks stored and indexed
```

### Step 2: Retrieval
When you ask a question:
```
Question: "What is mentioned about climate?"
    ↓
Search for relevant chunks (keyword or semantic)
    ↓
Return top 3 most relevant chunks
```

### Step 3: Generation
```
System: "You are a helpful assistant"
Context: "[Top 3 relevant chunks]"
Question: "What is mentioned about climate?"
    ↓
AI generates answer based on context
    ↓
Answer with source attribution
```

## 🎯 Key Concepts

### Chunking Strategy
```python
# Better chunking = better retrieval
- Chunk size: 500-1000 characters
- Overlap: 100-200 characters (preserve context)
- Split by: Sentences or paragraphs
```

### Retrieval Methods

**Keyword-Based** (Simple RAG)
- Fast, no embeddings needed
- Works: "Find chunks with matching words"
- Good for: Exact phrase matching

**Semantic Search** (Advanced RAG)
- More accurate understanding
- Works: "Find chunks with similar meaning"
- Good for: Conceptual matching, paraphrasing

### Prompt Engineering
```python
context = "[Retrieved document chunks]"
prompt = f"""
Based on this information:
{context}

Answer the question:
{question}
"""
```

## 🔧 Customization

### Change Retrieval Model
```python
rag = SimpleRAG(doc_paths=["your_doc.txt"])

# Use custom retrieval
chunks = rag._retrieve_relevant_chunks("question", top_k=5)
```

### Adjust Chunking
```python
rag._chunk_documents(chunk_size=1000, overlap=200)
# Larger chunks = more context but slower retrieval
# More overlap = better context preservation
```

### Change AI Model
```python
# In simple_rag.py, change:
self.model = "openrouter/auto"  # or any OpenRouter model
```

## 📊 Performance Tips

1. **Document Quality** - Clean, well-formatted documents → better results
2. **Chunk Size** - 500-1000 chars is usually optimal
3. **Top-K** - Retrieve 3-5 chunks (diminishing returns beyond)
4. **Query Clarity** - Specific questions → better results

## 🔒 Privacy & Security

- Documents are processed locally
- Only sent to OpenRouter API for generation
- API key stored in `.env` (never committed to git)
- No data stored on servers

## 🚀 Production Deployment

### For Web UI:
```bash
# Use Hugging Face Spaces, Gradio Cloud, or your server
python -m gunicorn rag_web_ui:app

# Or with Docker
docker build -t rag-app .
docker run -p 7860:7860 rag-app
```

### For API:
```python
from fastapi import FastAPI
from simple_rag import SimpleRAG

app = FastAPI()
rag = SimpleRAG()

@app.post("/ask")
def ask(question: str):
    return {"answer": rag.answer_question(question)}
```

## 📈 Enhancement Ideas

1. **Real Embeddings** - Use `sentence-transformers` or OpenAI embeddings
2. **Vector Database** - Store embeddings in Pinecone, Weaviate, or Milvus
3. **Multi-Document** - Support directories of documents
4. **Streaming** - Stream answers as they're generated
5. **Feedback Loop** - Learn from user feedback
6. **Hybrid Search** - Combine keyword + semantic search

## 🐛 Troubleshooting

**"No documents loaded"**
- Add .txt or .pdf files to the directory
- Check file names and paths

**"No relevant information found"**
- Try rephrasing your question
- Ensure documents contain relevant content
- Increase `top_k` for more chunks

**"API errors"**
- Check OPENROUTER_API_KEY in .env
- Verify internet connection
- Check OpenRouter service status

## 📚 References

- [RAG Techniques](https://arxiv.org/abs/2005.11401)
- [OpenRouter API](https://openrouter.ai/docs)
- [Gradio Docs](https://gradio.app/)
- [Document Chunking Best Practices](https://python.langchain.com/en/latest/modules/indexes/)

## 🎓 Next Steps

1. Try the simple RAG first
2. Add your own documents
3. Experiment with chunk sizes
4. Deploy the web UI
5. Upgrade to advanced RAG with real embeddings
6. Build production RAG system with vector database

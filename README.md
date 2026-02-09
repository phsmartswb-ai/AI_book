# 🤖 AI Document Intelligence Platform - RAG Chatbot

> **Ask questions about your documents. Get accurate answers with source citations.**

An enterprise-grade **Retrieval-Augmented Generation (RAG)** system that transforms your document collection into an intelligent Q&A assistant powered by AI.

[![GitHub](https://img.shields.io/badge/GitHub-phsmartswb--ai%2FAI__book-blue?logo=github)](https://github.com/phsmartswb-ai/AI_book)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ Key Features

- **🔍 Smart Document Retrieval** - Finds relevant information from your documents instantly
- **🤖 AI-Powered Answers** - Generates accurate responses based on actual document content
- **📚 Multi-Format Support** - Works with PDFs, text files, and mixed document collections
- **💬 Web Interface** - Beautiful Gradio UI with zero backend required
- **🔗 Multiple Retrieval Methods** - Keyword-based (fast) and semantic search (accurate)
- **📍 Source Attribution** - Every answer includes citations showing where information came from
- **🔐 Privacy First** - Documents processed locally, only sent to AI for generation

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🚀 Quick Start

### 1. Clone & Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/phsmartswb-ai/AI_book.git
cd AI_book

# Install dependencies
pip install -r requirements.txt

# Configure API key
echo "OPENROUTER_API_KEY=your-key-here" > .env
```

### 2. Run Web Interface

```bash
python rag_web_ui.py
```

Open browser to `http://localhost:7860` and start asking questions!

### 3. Example Questions

```
"What is the MVP?"
"What are the financial projections for year 1?"
"Summarize the city analysis findings"
"What is the unit economics breakdown?"
```

## 💾 Installation

### Requirements

- Python 3.8 or higher
- pip (Python package manager)
- OpenRouter API key

### Step-by-Step

```bash
# 1. Clone repository
git clone https://github.com/phsmartswb-ai/AI_book.git
cd AI_book

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file with API key
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-your-key-here
EOF

# 5. Run application
python rag_web_ui.py
```

### Get API Keys

1. **OpenRouter API** (Required)
   - Go to https://openrouter.ai/
   - Sign up and create API key
   - Add to `.env`

## 🎯 Usage

### Web Interface (Easiest)

```bash
python rag_web_ui.py
```

Then:
1. Type your question in the text box
2. Click "Get Answer"
3. View the AI response with source citations

### Python Script (Programmatic)

```python
from simple_rag import SimpleRAG

# Initialize with documents
rag = SimpleRAG(doc_paths=["MVP.pdf", "projections.pdf"])

# Ask a question
answer = rag.answer_question("What are the financial projections?")
print(answer)

# Interactive chat
rag.interactive_chat()
```

### Advanced RAG with Embeddings

```python
from advanced_rag import AdvancedRAG

# Initialize with semantic search
rag = AdvancedRAG(use_embeddings=True)

# Batch process questions
questions = [
    "What is the MVP?",
    "What are the projections?",
    "Who are the customers?"
]

results = rag.batch_qa(questions)
for qa in results:
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}\n")
```

## 🏗️ Architecture

### How RAG Works

```
User Question
    ↓
[RETRIEVE] → Search documents for relevant chunks
    ↓
[AUGMENT] → Build AI prompt with found context
    ↓
[GENERATE] → Send to AI model
    ↓
Answer with Source Citations
```

### System Components

| Component | Purpose | File |
|-----------|---------|------|
| **SimpleRAG** | Fast keyword-based retrieval | `simple_rag.py` |
| **AdvancedRAG** | Accurate semantic search | `advanced_rag.py` |
| **Web UI** | User-friendly interface | `rag_web_ui.py` |
| **Chatbots** | Alternative interfaces | `chatbot_*.py` |

### Document Processing

```
Input Documents
    ↓
Load (PDF/TXT parsing)
    ↓
Chunk (500-char chunks with overlap)
    ↓
Index (keyword/semantic indexing)
    ↓
Ready for retrieval
```

## 📚 Documentation

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** - Detailed project overview
- **[RAG_GUIDE.md](RAG_GUIDE.md)** - Complete RAG implementation guide
- **[requirements.txt](requirements.txt)** - All dependencies

## 💡 Examples

### Example 1: Ask About Financial Data

```python
rag = SimpleRAG(doc_paths=["projections.pdf", "UnitEconomics5Years.pdf"])

question = "What is the revenue projection for year 3?"
answer = rag.answer_question(question, verbose=True)
# Output includes retrieved chunks and source citations
```

### Example 2: Batch Analysis

```python
questions = [
    "What is the market size?",
    "Who are the main competitors?",
    "What is our competitive advantage?",
    "What is the go-to-market strategy?"
]

for q in questions:
    print(f"\nQ: {q}")
    answer = rag.answer_question(q)
    print(f"A: {answer}")
```

### Example 3: Document Upload

```python
# Via web UI:
# 1. Click "Upload Document"
# 2. Select PDF or TXT file
# 3. Click "Process Document"
# 4. Immediately available for querying

# Via code:
rag._load_document("new_document.pdf")
rag._chunk_documents()
answer = rag.answer_question("What's in the new document?")
```

## ⚙️ Configuration

### Environment Variables (.env)

```
# Required
OPENROUTER_API_KEY=your-key

```

### Customization

**Change retrieval method:**
```python
# Keyword-based (fast)
chunks = rag._retrieve_relevant_chunks(query, top_k=3)

# Semantic search (accurate)
chunks = rag._retrieve_by_similarity(query, top_k=3)
```

**Adjust chunk size:**
```python
rag._chunk_documents(chunk_size=1000, overlap=200)
```

**Use different AI model:**
```python
rag.model = "openrouter/openai/gpt-4"  # or any OpenRouter model
```

## 🔍 Retrieval Methods

### SimpleRAG - Keyword Matching
- ✅ Fast (sub-100ms)
- ✅ No ML model needed
- ✅ Good for exact phrases
- ❌ Misses semantic similarities

### AdvancedRAG - Semantic Search
- ✅ Accurate concept matching
- ✅ Handles paraphrasing
- ✅ Better relevance
- ⏱️ Slightly slower

## 🛠️ Troubleshooting

### "No documents loaded"
```bash
# Add documents to the directory
# Supported formats: .pdf, .txt

# Or specify paths explicitly
rag = SimpleRAG(doc_paths=["path/to/doc1.pdf", "path/to/doc2.txt"])
```

### "No relevant information found"
```python
# Try rephrasing question
# Increase top_k to retrieve more chunks
chunks = rag._retrieve_relevant_chunks(query, top_k=5)

# Check document content
for doc in rag.documents:
    print(f"Loaded: {doc['path']}")
```

### "OPENROUTER_API_KEY not set"
```bash
# Add to .env
echo "OPENROUTER_API_KEY=your-key" > .env

# Or set environment variable
export OPENROUTER_API_KEY=your-key
```

### "Gradio not installed"
```bash
pip install gradio
```

## 📊 Performance

| Metric | Value |
|--------|-------|
| Documents Indexed | 7+ |
| Max Chunks | 100+ |
| Retrieval Latency | <100ms |
| Generation Time | 2-5s |
| Accuracy (RAG) | High |
| Supported Formats | PDF, TXT |

## 🚀 Advanced Features

### Batch Processing
```python
rag = SimpleRAG()
questions = ["Q1", "Q2", "Q3"]
results = rag.batch_qa(questions)  # Available in AdvancedRAG
```

### Save Q&A Pairs
```python
qa_pairs = [
    {"question": "What is MVP?", "answer": "..."},
    {"question": "Projections?", "answer": "..."}
]
rag.save_qa_pairs(qa_pairs, "output.json")
```

## 🔐 Privacy & Security

- ✅ **Local Processing** - Documents processed on your machine
- ✅ **API-Only Sharing** - Only sent to OpenRouter for generation
- ✅ **No Data Storage** - No server-side storage of documents
- ✅ **Encrypted Transmission** - HTTPS for all API calls
- ✅ **.env Protection** - Never commit `.env` to version control

## 📈 Enhancement Roadmap

- [ ] Real embeddings (sentence-transformers)
- [ ] Vector database (Pinecone, Weaviate)
- [ ] REST API wrapper
- [ ] Mobile app interface
- [ ] Multi-language support
- [ ] Document version control
- [ ] Analytics dashboard
- [ ] Fine-tuned models

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🎓 Learn More

- [RAG Research Paper](https://arxiv.org/abs/2005.11401)
- [OpenRouter Docs](https://openrouter.ai/docs)
- [Gradio Tutorial](https://gradio.app/guides/)
- [PDF Processing](https://pypi.org/project/PyPDF2/)

## 📧 Contact

- **GitHub**: [phsmartswb-ai/AI_book](https://github.com/phsmartswb-ai/AI_book)
- **Issues**: [Report a bug](https://github.com/phsmartswb-ai/AI_book/issues)

## 🙏 Acknowledgments

- OpenRouter for flexible AI model access
- Gradio for beautiful web interfaces
- PyPDF2 for PDF processing
- Python community for excellent libraries

---

**Ready to get started?** Run `python rag_web_ui.py` and start asking questions! 🚀

Made with ❤️ for intelligent document analysis.

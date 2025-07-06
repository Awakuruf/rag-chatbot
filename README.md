# RAG Chatbot

A minimal Retrieval-Augmented Generation (RAG) chatbot that can answer questions based on your **PDF**, **Markdown**, and **web documents**. It uses OpenAI (or other LLMs) and includes a FastAPI backend to serve responses via a `/chat` endpoint.

---

## Project Structure

```
rag-chatbot/
│
├── app/
│ ├── main.py # FastAPI app with /chat endpoint
│ ├── rag_pipeline.py # Core RAG logic
│ ├── ingest.py # Loads + embeds PDF/Markdown/web docs
│
├── data/
│ ├── sample.pdf
│ ├── example.md
│ └── links.txt # One URL per line
│
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone the repo
```git clone https://github.com/your-username/rag-chatbot.git```
```cd rag-chatbot```

2. Install dependencies
 If pip doesn't work, try python -m pip install -r requirements.txt
```pip install -r requirements.txt```
1. Add your documents
Place .pdf and .md files in the data/ folder.

Add web URLs (one per line) to data/links.txt.

4. Ingest and embed documents
```python -m app.ingest```
This will embed your documents and store them using FAISS.

### 1. Start the API server
```uvicorn app.main:app --reload --port 8000```
1. Ask a question
```
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

### Features
- PDF, Markdown, and Web ingestion
- FAISS vector store for fast retrieval
- OpenAI-powered generation (can swap to local LLMs)
- FastAPI backend with /chat endpoint

### To-Do
 Add chat memory / session history

 Add frontend UI (React, Streamlit, etc.)

 Add support for more file types (DOCX, HTML, etc.)

 Swap OpenAI with local model (e.g., Mistral, LLaMA)

### License
MIT License. Use freely with credit appreciated.

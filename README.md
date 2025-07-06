# RAG-Bot: Daoist-Inspired Conversational Agent

A Retrieval-Augmented Generation (RAG) chatbot powered by Mistral-7B-Instruct, designed to deliver thoughtful, down-to-earth responses inspired by Daoist philosophy (Laozi, Zhuangzi). Built using FastAPI and optimized for lightweight, locally-hosted usage.

---

## Features

-  **RAG Architecture**: Retrieves relevant passages from a corpus of Daoist notes using semantic search with FAISS.
-  **Philosophy-Grounded Prompting**: Uses a system message that simulates a grounded Daoist mentor — gentle, honest, and reflective.
-  **Quantized Model Inference**: Runs Mistral-7B in 4-bit using bitsandbytes for faster generation on consumer GPUs.
-  **Dynamic Knowledge Base**: Loads and chunks `.md` notes from Daoist texts into embeddings at startup.
-  **API Endpoint**: Exposes a clean `POST /chat` route for frontend integration (e.g. Unity or web).

---

## Tech Stack

| Component              | Technology                          |
|------------------------|--------------------------------------|
| Language Model         | `mistralai/Mistral-7B-Instruct-v0.2` |
| Embeddings             | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Search          | `FAISS`                             |
| API Framework          | `FastAPI`                           |
| Quantization           | `bitsandbytes` (4-bit NF4)          |
| Tokenizer & Model      | `transformers`                      |

---

## Quickstart

### Clone the Repo

```bash
git clone https://github.com/Awakuruf/rag-chatbot.git
cd rag-bot
```

### Make sure your backend is running before pressing Play in Unity.

1.Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the FastAPI server:

```bash
cd .\app\
uvicorn main:app --reload
```

4. The Unity game will POST messages to http://127.0.0.1:8000/chat.
   
## Folder Structure
```
rag-chatbot/
│
├── app/
│ ├── main.py # FastAPI app with /chat endpoint
│ ├── rag_pipeline.py # Core RAG logic
│ ├── ingest.py # Loads + embeds PDF/Markdown/web docs
│
├── data/
│ ├── daodejing.pdf
│ ├── daodejing_notes.md
| ├── zhuangzi.pdf
│ └── zhuangzi_notes.md
│
├── requirements.txt
├── example_responses.txt
└── README.md
```

## Troubleshooting
- Long response times from AI?
Consider:

    - Reducing max_new_tokens
    - Using smaller models like mistral-7b-instruct in 4-bit mode
    - Chunking your documents more efficiently

## License
MIT License. Feel free to remix or adapt for educational and non-commercial purposes.

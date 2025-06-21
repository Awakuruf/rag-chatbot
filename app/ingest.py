from langchain.document_loaders import PyPDFLoader, WebBaseLoader, MarkdownLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os

def load_docs():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            docs += PyPDFLoader(f"data/{file}").load()
        elif file.endswith(".md"):
            docs += MarkdownLoader(f"data/{file}").load()
    with open("data/links.txt", "r") as f:
        urls = f.read().splitlines()
    docs += WebBaseLoader(urls).load()
    return docs

def create_vector_store():
    docs = load_docs()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store created.")
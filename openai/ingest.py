from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    UnstructuredMarkdownLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_docs():
    docs = []
    for file in os.listdir("data"):
        if file.endswith(".pdf"):
            docs += PyPDFLoader(f"data/{file}").load()
        elif file.endswith(".md"):
            docs += UnstructuredMarkdownLoader(f"data/{file}").load()
    
    # Web links
    links_file = "data/links.txt"
    if os.path.exists(links_file):
        with open(links_file, "r") as f:
            urls = f.read().splitlines()
        os.environ["USER_AGENT"] = "rag-bot/0.1"
        docs += WebBaseLoader(urls).load()
    
    return docs

def create_vector_store():
    docs = load_docs()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vector store created.")

if __name__ == "__main__":
    create_vector_store()

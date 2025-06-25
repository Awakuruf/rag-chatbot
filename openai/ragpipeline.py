from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

def load_qa_chain():
    vectorstore = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    llm = OpenAI()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

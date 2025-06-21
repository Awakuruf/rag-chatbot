from fastapi import FastAPI, Request
from app.rag_pipeline import load_qa_chain

app = FastAPI()
qa_chain = load_qa_chain()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question", "")
    answer = qa_chain.run(question)
    return {"answer": answer}

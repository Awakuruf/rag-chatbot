from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import generate_response, add_documents

app = FastAPI()

# Initial Daoist texts (load more in practice)
daoist_knowledge = [
    "The Dao that can be told is not the eternal Dao. The name that can be named is not the eternal name.",
    "To know that you do not know is the best. To pretend to know when you do not know is a disease.",
    "Those who know do not speak. Those who speak do not know."
]

add_documents(daoist_knowledge)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(input: ChatInput):
    response = generate_response(input.message)
    return {"response": response}

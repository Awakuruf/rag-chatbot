import markdown
import os

from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import generate_response, add_documents

app = FastAPI()

# Initial Daoist texts (load more in practice)
# daoist_knowledge = [
#     "The Dao that can be told is not the eternal Dao. The name that can be named is not the eternal name.",
#     "To know that you do not know is the best. To pretend to know when you do not know is a disease.",
#     "Those who know do not speak. Those who speak do not know."
# ]

def load_and_chunk_notes(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks

BASE_DIR = os.path.dirname(__file__)
NOTES_PATH = os.path.join(BASE_DIR, "..", "data", "daodejing_notes.md")
NOTES_PATH_2 = os.path.join(BASE_DIR, "..", "data", "zhuangzi_notes.md")
daoist_knowledge = load_and_chunk_notes(NOTES_PATH)
daoist_knowledge2 = load_and_chunk_notes(NOTES_PATH_2)

add_documents(daoist_knowledge)
add_documents(daoist_knowledge2)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat(input: ChatInput):
    response = generate_response(input.message)
    return {"response": response}

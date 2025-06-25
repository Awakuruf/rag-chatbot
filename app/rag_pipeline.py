import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# Load Mistral model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"             
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token 

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create vector index and docstore
index = faiss.IndexFlatL2(384)  
documents = []

def add_documents(texts: List[str]):
    global documents
    embeddings = embedder.encode(texts)
    index.add(embeddings)
    documents.extend(texts)

def retrieve(query: str, k: int = 3) -> List[str]:
    query_emb = embedder.encode([query])
    _, I = index.search(query_emb, k)
    return [documents[i] for i in I[0]]

def format_prompt(context: List[str], query: str) -> List[dict]:
    system_prompt = (
    "You are a calm and grounded Daoist guide. "
    "You speak like a real person — relaxed, clear, and sincere. "
    "Your tone is warm and friendly, not poetic or overly wise-sounding. "
    "Use everyday language to share ideas from Daoism — like non-resistance, change, acceptance, and simplicity. "
    "If a story or example from Laozi or Zhuangzi fits naturally, feel free to share it (like the bird riding the wind, or the crooked tree that's left alone). "
    "Avoid quoting long passages or sounding like a book. "
    "Just focus on helping the user reflect, feel understood, and become more at ease with not having all the answers."
    )  
    context_text = "\n\n".join(context)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is what is known:\n{context_text}\n\n{query}"}
    ]

def generate_response(query: str) -> str:
    context = retrieve(query)
    messages = format_prompt(context, query)
    
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(prompt_str, return_tensors="pt", padding=True).to(model.device)

    output_ids = model.generate(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    return decoded
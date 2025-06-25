import faiss
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from typing import List

from stopping_criteria import StopOnDoubleNewline

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
    "You're a grounded and friendly Daoist guide. "
    "You speak like a real person — calm, honest, and down-to-earth. "
    "Use everyday language to help people reflect and feel okay with uncertainty. "
    "You draw from ideas in Daoism (like change, letting go, and simplicity), but avoid sounding poetic or like you're giving a lecture. "
    "If it's helpful, you can casually bring up stories or metaphors from Laozi or Zhuangzi — but only if it fits the moment naturally."
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

    input_len = tokenized["input_ids"].shape[-1]

    # stopping_criteria = StoppingCriteriaList([
    #     StopOnDoubleNewline(tokenizer, start_length=input_len)
    # ])
    
    output_ids = model.generate(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "[/INST]" in decoded:
        decoded = decoded.split("[/INST]")[-1].strip()

    return truncate_to_last_full_sentence(decoded)

def truncate_to_last_full_sentence(text: str) -> str:
    # Find the last sentence-ending punctuation
    matches = list(re.finditer(r'[.!?]["\']?\s', text))
    if matches:
        last_end = matches[-1].end()
        return text[:last_end].strip()
    return text.strip()

# Example Response

# {
#     "response": "Hey there, I hear you're feeling a bit uneasy about the constant changes in life, huh? 
#                   That's totally normal – change is something we all experience, and it can be tough to find peace in the midst of it.
#                   But let me share a little something I learned from the ancient Daoist wisdom, which might help you find some inner peace. 
#                   I'd like to paraphrase a famous quote from Laozi, the ancient Daoist sage:
#                   "Those who know do not speak, and those who speak do not know.
#                   This quote might seem a bit confusing at first, but what it's getting at is that sometimes, the best way to understand something is to let go of trying to grasp it with your mind. 
#                   When we're too attached to our thoughts and ideas about how things should be, we can miss out on the actual experience of the moment.
#                   Another quote from Laozi that comes to mind is,"To know that you do not know is the best.\" This means acknowledging that we don't have all the answers and that it's okay not to have everything figured out. 
#                   Pretending to know when we don't can actually cause"
# }
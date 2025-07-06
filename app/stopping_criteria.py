import re
import torch
from transformers import StoppingCriteria

class StopOnDoubleNewline(StoppingCriteria):
    def __init__(self, tokenizer, start_length: int, min_tokens: int = 50):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.min_tokens = min_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_tokens = input_ids[0][self.start_length:]
        if len(generated_tokens) < self.min_tokens:
            return False  
        
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Stop only if double newline appears and there's a decent amount of content
        if "\n\n" in decoded or re.search(r"[\.!?]\s*\n", decoded):
            return True

        return False

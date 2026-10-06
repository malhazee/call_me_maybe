import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

def extract_number(base_prefix: str, model: Small_LLM_Model) -> float:
    # Constrain to digits 0-9 and decimal point
    digits = [str(i) for i in range(10)] + [".", ",", "}"]
    allowed_tokens = {}
    for ch in digits:
        tok_id = model.encode(ch)[0].tolist()[-1]
        allowed_tokens[tok_id] = ch
    
    current_ids = model.encode(base_prefix)[0].tolist()
    num_str = ""
    for _ in range(10): # max 10 chars
        logits = model.get_logits_from_input_ids(current_ids)
        # pick best among allowed
        best_tok = max(allowed_tokens.keys(), key=lambda t: logits[t])
        ch = allowed_tokens[best_tok]
        if ch in [",", "}"]:
            break
        num_str += ch
        current_ids.append(best_tok)
    try:
        return float(num_str)
    except:
        return 0.0

prefix = '{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": '
val_a = extract_number(prefix, model)
print("Extracted a:", val_a)

prefix_b = f'{prefix}{val_a}, "b": '
val_b = extract_number(prefix_b, model)
print("Extracted b:", val_b)

from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

def extract_string(base_prefix: str, model: Small_LLM_Model) -> str:
    current_ids = model.encode(base_prefix)[0].tolist()
    quote_id = model.encode('"')[0].tolist()[-1]
    result_str = ""
    for _ in range(20): # max 20 tokens
        logits = model.get_logits_from_input_ids(current_ids)
        # pick argmax token, but ignore space if at start or pick top
        # if the model picks quote, we stop!
        best_tok = max(range(len(logits)), key=lambda t: logits[t])
        if best_tok == quote_id:
            break
        text = model.decode([best_tok])
        if '"' in text:
            result_str += text.split('"')[0]
            break
        result_str += text
        current_ids.append(best_tok)
    return result_str

prefix = '{"prompt": "Greet shrek", "name": "fn_greet", "parameters": {"name": "'
name_val = extract_string(prefix, model)
print("Extracted name:", name_val)

import json
from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
with open("data/input/functions_definition.json", "r") as f:
    funcs_data = json.load(f)
available_functions = [fn["name"] for fn in funcs_data]

prompts = [
    "What is the sum of 2 and 3?",
    "Greet shrek",
    "Calculate the square root of 144",
]

print("--- Test 1: RAW PROMPT ONLY (No formatted input) ---")
for prompt in prompts:
    base = model.encode(prompt)[0].tolist()
    scores = {}
    for fn in available_functions:
        fn_toks = model.encode(fn)[0].tolist()
        cur = list(base)
        s = 0.0
        for t in fn_toks:
            logits = model.get_logits_from_input_ids(cur)
            s += logits[t]
            cur.append(t)
        scores[fn] = s / len(fn_toks)
    print(f"'{prompt}' -> Best: {max(scores, key=scores.get)}")

print("\n--- Test 2: JSON PREFIX (The real project format!) ---")
for prompt in prompts:
    prefix = f'{{"prompt": "{prompt}", "name": "'
    base = model.encode(prefix)[0].tolist()
    scores = {}
    for fn in available_functions:
        fn_toks = model.encode(fn)[0].tolist()
        cur = list(base)
        s = 0.0
        for t in fn_toks:
            logits = model.get_logits_from_input_ids(cur)
            s += logits[t]
            cur.append(t)
        scores[fn] = s / len(fn_toks)
    print(f"'{prompt}' -> Best: {max(scores, key=scores.get)}")

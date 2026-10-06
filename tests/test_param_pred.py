from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
prefix = '{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": '
ids = model.encode(prefix)[0].tolist()
logits = model.get_logits_from_input_ids(ids)

digits = [str(i) for i in range(10)]
digit_scores = {}
for d in digits:
    d_tok = model.encode(d)[0].tolist()[-1]
    digit_scores[d] = logits[d_tok]

print("Prefix:", prefix)
print("Digit scores for 'a':")
for d, score in sorted(digit_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  Digit '{d}': {score:.2f}")
print("Best digit chosen for 'a':", max(digit_scores, key=digit_scores.get))

from typing import Any, Dict, List
import numpy as np
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult


def sample_masked_token(
        logits_list: List[float],
        valid_token_ids: List[int],
) -> int:
    logits = np.array(logits_list, dtype=np.float32)
    mask = np.ones(len(logits), dtype=bool)
    mask[valid_token_ids] = False
    logits[mask] = -np.inf
    return int(np.argmax(logits))


def select_best_function(
        prompt: str,
        availavle_functions: List[str],
        model: Small_LLM_Model,
) -> str:
    empty_prefix = '{"prompt": "", "name": "'
    empty_base = model.encode(empty_prefix)[0].tolist()

    curr_prefix = f'{{"prompt": "{prompt}", "name": "'
    curr_base = model.encode(curr_prefix)[0].tolist()

    scores: Dict[str, float] = {}
    for fn_name in availavle_functions:
        fn_tokens = model.encode(fn_name)[0].tolist()
        num_toks = len(fn_tokens)

        # 1. Baseline prior score (with empty prompt)
        empty_curr = list(empty_base)
        base_score = 0.0
        for token_id in fn_tokens:
            logits = model.get_logits_from_input_ids(empty_curr)
            base_score += logits[token_id]
            empty_curr.append(token_id)
        baseline = base_score / num_toks

        # 2. Actual conditional score (with user prompt)
        curr_tokens = list(curr_base)
        total_score = 0.0
        for token_id in fn_tokens:
            logits = model.get_logits_from_input_ids(curr_tokens)
            total_score += logits[token_id]
            curr_tokens.append(token_id)
        norm_score = total_score / num_toks

        # Calibrated score (PMI with alpha = 0.6)
        scores[fn_name] = norm_score - (0.6 * baseline)

    return max(scores, key=lambda k: scores[k])


def extract_number_param(
        prefix: str,
        model: Small_LLM_Model,
        is_integer: bool = False,
) -> float | int:
    digits = [str(i) for i in range(10)] + [".", "-", ",", "}"]
    allowed_map = {model.encode(ch)[0].tolist()[-1]: ch for ch in digits}
    valid_ids = list(allowed_map.keys())

    current_ids = model.encode(prefix)[0].tolist()
    num_str = ""
    for _ in range(12):
        logits = model.get_logits_from_input_ids(current_ids)
        best_tok = sample_masked_token(logits, valid_ids)
        ch = allowed_map[best_tok]
        if ch in [",", "}"]:
            break
        num_str += ch
        current_ids.append(best_tok)

    try:
        val = float(num_str)
        return int(val) if is_integer else val
    except ValueError:
        return 0 if is_integer else 0.0


def extract_string_param(prefix: str, model: Small_LLM_Model) -> str:
    current_ids = model.encode(prefix)[0].tolist()
    quote_id = model.encode('"')[0].tolist()[-1]
    result_str = ""

    for _ in range(30):
        logits = model.get_logits_from_input_ids(current_ids)
        best_tok = int(np.argmax(logits))
        if best_tok == quote_id:
            break
        text = model.decode([best_tok])
        if '"' in text:
            result_str += text.split('"')[0]
            break
        result_str += text
        current_ids.append(best_tok)

    return result_str.strip()


def generate_function_call(
        prompt: str,
        functions_meta: Dict[str, FunctionDefinition],
        model: Small_LLM_Model,
) -> FunctionCallResult:
    available_functions = list(functions_meta.keys())
    fn_name = select_best_function(prompt, available_functions, model)
    fn_def = functions_meta[fn_name]

    parameters: Dict[str, Any] = {}
    running_prefix = (
        f'User request: "{prompt}"\n'
        f'Call: {{"name": "{fn_name}", "parameters": {{'
    )

    param_items = list(fn_def.parameters.items())
    for idx, (p_name, p_spce) in enumerate(param_items):
        running_prefix += f'"{p_name}": '

        if p_spce.type in ("number", "float", "integer", "int"):
            is_int = p_spce.type in ("integer", "int")
            num_val = extract_number_param(
                running_prefix, model, is_integer=is_int
            )
            parameters[p_name] = num_val
            running_prefix += str(num_val)
        else:
            running_prefix += '"'
            str_val = extract_string_param(running_prefix, model)
            parameters[p_name] = str_val
            running_prefix += f'{str_val}"'

        if idx < len(param_items) - 1:
            running_prefix += ", "

    running_prefix += "}}"

    return FunctionCallResult(
        prompt=prompt,
        name=fn_name,
        parameters=parameters,
    )

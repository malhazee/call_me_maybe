"""Constrained decoding engine for function calling."""
from typing import Any, Dict, List
import numpy as np
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, FunctionCallResult


def sample_masked_token(
    logits_list: List[float],
    valid_token_ids: List[int],
) -> int:
    """Mask invalid token logits to negative infinity and select argmax.

    Following Subject Section 5.3.3:
    1. Produces logits for all possible tokens.
    2. Identifies valid tokens maintaining valid JSON and schema.
    3. Sets logits for invalid tokens to negative infinity (-inf).
    4. Samples from the remaining valid tokens (Greedy argmax).

    Args:
        logits_list: Raw unnormalized logits from the LLM model.
        valid_token_ids: List of allowed token IDs for this step.

    Returns:
        The chosen token ID among valid candidates with highest score.
    """
    logits = np.array(logits_list, dtype=np.float32)
    mask = np.ones(len(logits), dtype=bool)
    mask[valid_token_ids] = False
    logits[mask] = -np.inf
    return int(np.argmax(logits))


def select_best_function(
    prompt: str,
    available_functions: List[str],
    model: Small_LLM_Model,
) -> str:
    """Select the best function name using length-normalized scoring.

    Args:
        prompt: Natural language user prompt.
        available_functions: List of available function names.
        model: Small_LLM_Model instance.

    Returns:
        The function name with the highest normalized logit score.
    """
    prefix = f'{{"prompt": "{prompt}", "name": "'
    base_tokens = model.encode(prefix)[0].tolist()

    scores: Dict[str, float] = {}
    for fn_name in available_functions:
        fn_tokens = model.encode(fn_name)[0].tolist()
        current_tokens = list(base_tokens)
        total_score = 0.0
        for token_id in fn_tokens:
            logits = model.get_logits_from_input_ids(current_tokens)
            total_score += logits[token_id]
            current_tokens.append(token_id)
        scores[fn_name] = total_score / len(fn_tokens)

    return max(scores, key=lambda k: scores[k])


def extract_number_param(prefix: str, model: Small_LLM_Model) -> float:
    """Extract a number parameter under strict digit constraints.

    Uses negative-infinity masking to only allow valid number characters.

    Args:
        prefix: Current running JSON string prefix.
        model: Small_LLM_Model instance.

    Returns:
        Extracted floating-point value.
    """
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
        return float(num_str)
    except ValueError:
        return 0.0


def extract_string_param(prefix: str, model: Small_LLM_Model) -> str:
    """Extract a string parameter by generating tokens until the closing quote.

    Args:
        prefix: Current running JSON string prefix.
        model: Small_LLM_Model instance.

    Returns:
        Extracted and cleaned string value.
    """
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
    """Orchestrate the full constrained decoding pipeline for a given prompt.

    Args:
        prompt: Natural language user prompt.
        functions_meta: Mapping of function names to FunctionDefinitions.
        model: Small_LLM_Model instance.

    Returns:
        A validated FunctionCallResult model.
    """
    available_functions = list(functions_meta.keys())
    fn_name = select_best_function(prompt, available_functions, model)
    fn_def = functions_meta[fn_name]

    parameters: Dict[str, Any] = {}
    running_prefix = (
        f'{{"prompt": "{prompt}", "name": "{fn_name}", "parameters": {{'
    )

    param_items = list(fn_def.parameters.items())
    for idx, (p_name, p_spec) in enumerate(param_items):
        running_prefix += f'"{p_name}": '

        if p_spec.type == "number":
            num_val = extract_number_param(running_prefix, model)
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

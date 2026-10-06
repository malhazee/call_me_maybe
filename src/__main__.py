import argparse
import json
import os
import sys
from typing import Dict, List
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition, PromptInput
from src.decoder import generate_function_call


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Function Calling with Constrained Decoding",
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file defining available functions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output JSON file where results will be saved.",
    )
    return parser.parse_args()


def load_functions(file_path: str) -> Dict[str, FunctionDefinition]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Functions file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(
            "Functions definition file must contain a JSON array."
        )

    functions_map: Dict[str, FunctionDefinition] = {}
    for item in raw_data:
        fn_def = FunctionDefinition.model_validate(item)
        functions_map[fn_def.name] = fn_def

    return functions_map


def load_test_prompts(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Input file must contain a JSON array of prompts.")

    prompts: List[str] = []
    for item in raw_data:
        prompt_item = PromptInput.model_validate(item)
        prompts.append(prompt_item.prompt)

    return prompts


def main() -> None:
    args = parse_arguments()
    try:
        functions_meta = load_functions(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as err:
        print(f"Error loading input files: {err}", file=sys.stderr)
        sys.exit(1)

    print(f"loaded {len(functions_meta)} functions.")
    print(f"Processing {len(prompts)} test prompts...\n")

    try:
        model = Small_LLM_Model()
    except Exception as err:
        print(f"Error initializing LLM model: {err}", file=sys.stderr)
        sys.exit(1)

    result = []
    for idx, prompt in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] Processing: '{prompt}'")
        call_result = generate_function_call(prompt, functions_meta, model)
        result.append(call_result.model_dump())

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nSuccessfully processed all prompts!")
    print(f"results written to: {args.output}")


if __name__ == "__main__":
    main()

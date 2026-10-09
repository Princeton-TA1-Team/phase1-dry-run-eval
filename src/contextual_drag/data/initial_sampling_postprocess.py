from collections import Counter
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
_HARMONY_END_TOKENS = ("<|return|>", "<|end|>", "<|endoftext|>")
_GEMMA_THINKING_END = "<channel|>"
_GEMMA_THINKING_START = "<|channel>"


def parse_thinking_steps(response: str, prompt: str, max_response_length: int):
    """
    Split a response into its final, answer-facing text and a status.

    Example:
        >>> parse_thinking_steps('<|channel|>analysis<|message|>hm<|end|><|start|>assistant<|channel|>final<|message|>Ans<|return|>', '', 100)
        ('Ans', 'parsable_thinking')
        >>> parse_thinking_steps('<|channel>thought hm<channel|>Ans', '', 100)
        ('Ans', 'parsable_thinking')
        >>> parse_thinking_steps('<|channel>thought hm', '', 100)
        ('', 'malformed_thinking')
    """
    # gpt-oss harmony with special tokens kept (skip_special_tokens=False,
    # which is how this package samples).
    if _HARMONY_FINAL in response:
        final_part = response.split(_HARMONY_FINAL)[-1]
        for tok in _HARMONY_END_TOKENS:
            if final_part.endswith(tok):
                final_part = final_part[: -len(tok)]
        return final_part.strip(), 'parsable_thinking'
    if _HARMONY_ANALYSIS in response:
        return response, 'malformed_thinking'

    # Gemma 4: reasoning in a channel closed by <channel|>. An unclosed
    # channel has no final text, as in the original contextual-drag parser.
    if _GEMMA_THINKING_END in response:
        return response.rsplit(_GEMMA_THINKING_END, 1)[-1].lstrip(), 'parsable_thinking'
    if response.lstrip().startswith(_GEMMA_THINKING_START):
        return '', 'malformed_thinking'

    if response.startswith("analysis") and response[8] != " ":
        # In GPT-OSS format
        if "assistantfinal" in response:
            non_thinking_response = response.split("assistantfinal")[-1]
            thinking_status = 'parsable_thinking'
            return non_thinking_response, thinking_status
        
        else:
            non_thinking_response = response
            thinking_status = 'malformed_thinking'
            return non_thinking_response, thinking_status

    if "<think>" not in prompt + response:
        non_thinking_response = response
        thinking_status = 'no_thinking'
        return non_thinking_response, thinking_status
    
    else:
        non_thinking_response = response.split("</think>")[-1]
        concatenated_response = prompt + response
        if concatenated_response.count("<think>") != concatenated_response.count("</think>"):
            thinking_status = 'malformed_thinking'
        else:
            thinking_status = 'parsable_thinking'
    
    if len(non_thinking_response) > max_response_length and thinking_status != 'parsable_thinking':
        non_thinking_response = non_thinking_response[:max_response_length]
        thinking_status = 'truncated_' + thinking_status
        # print(f"Truncated {thinking_status} response to {len(non_thinking_response)} characters")
    
    return non_thinking_response, thinking_status

def preprocess_entry(entry, max_response_length):

    necessary_keys = ['problem', 'answer', 'init_response_generations_generated_response', 'id', 'init_response_prompt']
    for key in necessary_keys:
        if key not in entry:
            raise ValueError(f"Missing key: {key}")

    # Further processing can be done here
    final_response, thinking_status = parse_thinking_steps(entry['init_response_generations_generated_response'], entry['init_response_prompt'], max_response_length)
    entry['init_response_final'] = final_response
    entry['init_response_thinking_status'] = thinking_status
    entry['response_unique_id'] = get_unique_traj_id(entry)
    return entry

def get_unique_traj_id(entry):
    return "-".join([str(s) for s in [
        entry['id'],
        entry['init_response_generations_metadata']['model_config_alias'],
        entry['init_response_generations_response_id']
    ]])

def main(args):
    input_dir = Path(args.input_dir)
    max_response_length = args.max_response_length
    output_file = input_dir / "processed_flattened_init_responses.ds"

    print(f"\nPreprocessing dataset with max response length {max_response_length}")

    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist!")
        return 1

    all_files = glob(str(input_dir / args.input_file_template))
    if not all_files:
        print(f"ERROR: No dataset files found in '{input_dir}'!")
        return 1

    processed_entries = []
    for file_path in all_files:
        print(f"\nProcessing file: {file_path}")

        total_entries = 0
        thinking_parsing_status = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_entries += 1
                entry = json.loads(line)
                processed_entry = preprocess_entry(entry, max_response_length)
                processed_entries.append(processed_entry)

                thinking_parsing_status.append(processed_entry['init_response_thinking_status'])

        # Compute the percentage of entries with each thinking parsing status
        thinking_parsing_status_counts = Counter(thinking_parsing_status)
        total_entries = sum(thinking_parsing_status_counts.values())
        thinking_parsing_status_percentages = {status: count / total_entries * 100 for status, count in thinking_parsing_status_counts.items()}
        max_status_len = max(len(s) for s in thinking_parsing_status_percentages)
        for status, percentage in thinking_parsing_status_percentages.items():
            print(f"{status:<{max_status_len}} : {percentage:6.2f}%")

    # Save the processed entries to a hf dataset
    dataset = Dataset.from_list(processed_entries)
    dataset.save_to_disk(str(output_file))

    print(f"Preprocessing complete. Processed {len(processed_entries)} entries.")
    print(f"Output saved to '{output_file}'.")
    return 0

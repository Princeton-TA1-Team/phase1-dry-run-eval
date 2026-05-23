from collections import Counter
import json
from glob import glob
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

def parse_thinking_steps(response: str, prompt: str, max_response_length: int):

    if response.startswith("analysis") and response[7] != " ":
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

def preprocess_entry(entry, max_response_length, r):

    if r == 0:
        necessary_keys = ['problem', 'answer', 'init_response_generations_generated_response', 'id', 'init_response_prompt']
    else:
        necessary_keys = ['problem', 'answer', f'round{r}_response_generations_generated_response', 'id', f'round{r}_response_prompt']
    for key in necessary_keys:
        if key not in entry:
            raise ValueError(f"Missing key: {key}")

    # Further processing can be done here
    if r == 0:
        final_response, thinking_status = parse_thinking_steps(entry['init_response_generations_generated_response'], entry['init_response_prompt'], max_response_length)
        entry['init_response_final'] = final_response
        entry['init_response_thinking_status'] = thinking_status
    else:
        final_response, thinking_status = parse_thinking_steps(entry[f'round{r}_response_generations_generated_response'], entry[f'round{r}_response_prompt'], max_response_length)
        entry[f'round{r}_response_final'] = final_response
        entry[f'round{r}_response_thinking_status'] = thinking_status
        entry['response_unique_id'] = get_unique_traj_id(entry, r)
    return entry

def get_unique_traj_id(entry, r):
    if r == 0:
        return "-".join([str(s) for s in [
            entry['id'],
            entry['init_response_generations_metadata']['model_config_alias'],
            entry['init_response_generations_response_id']
        ]])
    else:
        return "-".join([str(s) for s in [
            entry['id'],
            entry[f'round{r}_response_generations_metadata']['model_config_alias'],
            entry[f'round{r}_response_generations_response_id'],
            f'round{r}'
        ]])

def main(args):
    input_dir = Path(args.input_dir)
    max_response_length = args.max_response_length
    if args.output_dir:
        output_file = Path(args.output_dir) / f"processed_flattened_round{args.round_num}_responses.ds"
    else:
        output_file = input_dir / f"processed_flattened_round{args.round_num}_responses.ds"
    r = args.round_num

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
                processed_entry = preprocess_entry(entry, max_response_length, r)
                processed_entries.append(processed_entry)

                if r == 0:
                    thinking_parsing_status.append(processed_entry['init_response_thinking_status'])
                else:
                    thinking_parsing_status.append(processed_entry[f'round{r}_response_thinking_status'])

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

import json
from datasets import load_from_disk
import copy
import os

def preprocess_api_data(args):
    if args.data_format == 'openai_api':
        return preprocess_openai_api_data(args)
    elif args.data_format == 'gemini_api':
        return preprocess_gemini_api_data(args)
    else:
        raise ValueError(f"Invalid data format: {args.data_format}")

def preprocess_openai_api_data(args):
    print("Preprocessing OpenAI API data")
    # First load the jsonl raw file
    assert args.dataset_dir.endswith('.jsonl'), "Dataset directory must end with .jsonl for OpenAI API data preprocessing"
    raw_api_data = []

    benchmark_ds = None
    metadict = {}

    with open(args.dataset_dir, 'r') as f:
        for line in f:
            item = json.loads(line)
            raw_api_data.append(item)
            benchmark_name = item['custom_id'].split('_')[0]
            problem_id = item['custom_id'].split('_')[1].split('-')[0]
            generation_id = item['custom_id'].split('_')[1].split('-')[1]

            if problem_id not in metadict:
                metadict[problem_id] = {}
            
            metadict[problem_id][generation_id] = item

            if benchmark_ds is None:
                print(f"Loading problem data for {benchmark_name}")
                benchmark_ds = load_from_disk(args.problem_data_path_root.format(benchmark_name=benchmark_name))

    # Then load the problem data
    new_api_data = []
    for problem_item in benchmark_ds:
        new_item = copy.deepcopy(problem_item)
        problem_id = problem_item['id']
        generation_ls = []
        if problem_id not in metadict:
            print(f"Problem ID {problem_id} not found in metadict")
            continue
        for generation_id in metadict[problem_id]:
            generated_meta = metadict[problem_id][generation_id]
            generation_ls.append({
                'generated_response': generated_meta['response']['body']['choices'][0]['message']['content'],
                'finish_reason': generated_meta['response']['body']['choices'][0]['finish_reason'],
                'metadata': generated_meta
            })
        new_item['init_response_generations'] = generation_ls
        new_api_data.append(new_item)

    args.dataset_dir = os.path.dirname(args.dataset_dir)
    print(f"args.dataset_dir: {args.dataset_dir}")

    return new_api_data


def preprocess_gemini_api_data(args):
    print("Preprocessing Gemini API data")

    # First load the jsonl raw file
    assert args.dataset_dir.endswith('.jsonl'), "Dataset directory must end with .jsonl for OpenAI API data preprocessing"
    raw_api_data = []

    benchmark_ds = None
    metadict = {}

    with open(args.dataset_dir, 'r') as f:
        for line in f:
            item = json.loads(line)
            raw_api_data.append(item)
            benchmark_name = item['key'].split('_')[0]
            problem_id = item['key'].split('_')[1].split('-')[0]
            generation_id = item['key'].split('_')[1].split('-')[1]

            if problem_id not in metadict:
                metadict[problem_id] = {}
            
            metadict[problem_id][generation_id] = item

            if benchmark_ds is None:
                print(f"Loading problem data for {benchmark_name}")
                benchmark_ds = load_from_disk(args.problem_data_path_root.format(benchmark_name=benchmark_name))

    # Then load the problem data
    new_api_data = []
    for problem_item in benchmark_ds:
        new_item = copy.deepcopy(problem_item)
        problem_id = problem_item['id']
        generation_ls = []
        if problem_id not in metadict:
            print(f"Problem ID {problem_id} not found in metadict")
            continue
        for generation_id in metadict[problem_id]:
            generated_meta = metadict[problem_id][generation_id]
            generation_ls.append({
                'generated_response': generated_meta['response']['candidates'][0]['content']['parts'][0]['text'],
                'finish_reason': generated_meta['response']['candidates'][0]['finishReason'],
                'metadata': generated_meta
            })

        new_item['init_response_generations'] = generation_ls
        new_api_data.append(new_item)

    args.dataset_dir = os.path.dirname(args.dataset_dir)
    print(f"args.dataset_dir: {args.dataset_dir}")

    return new_api_data

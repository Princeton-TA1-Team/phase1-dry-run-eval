"""
Dataset utilities for loading and validating datasets.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from tqdm import tqdm


def check_dataset_completeness(dataset_dir: str) -> Tuple[bool, List[str]]:
    """
    Check if all partition files exist and are complete.
    Returns (is_complete, missing_files).
    """
    dataset_path = Path(dataset_dir)
    
    # Find all dataset files
    dataset_files = list(dataset_path.glob("dataset-*of*.jsonl"))
    
    if not dataset_files:
        return False, [f"No dataset files found in {dataset_dir}"]
    
    # Extract partition information
    partition_info = []
    for file_path in dataset_files:
        filename = file_path.name
        match = re.match(r'dataset-(\d+)of(\d+)\.jsonl', filename)
        if match:
            partition_id = int(match.group(1))
            total_partitions = int(match.group(2))
            partition_info.append((partition_id, total_partitions, file_path))
    
    if not partition_info:
        return False, [f"Could not parse partition information from files in {dataset_dir}"]
    
    # Check if we have all partitions
    total_partitions = partition_info[0][1]  # All should have same total
    expected_partitions = set(range(total_partitions))
    found_partitions = set(info[0] for info in partition_info)
    
    missing_partitions = expected_partitions - found_partitions
    missing_files = []
    
    if missing_partitions:
        for partition_id in missing_partitions:
            missing_files.append(f"dataset-{partition_id}of{total_partitions}.jsonl")
    
    # Check if files are not empty
    empty_files = []
    for partition_id, total_partitions, file_path in partition_info:
        if file_path.stat().st_size == 0:
            empty_files.append(file_path.name)
    
    missing_files.extend(empty_files)
    
    is_complete = len(missing_files) == 0
    
    return is_complete, missing_files


def load_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Load and combine all partition files into a unified dataset.
    """
    dataset_path = Path(dataset_dir)
    dataset_files = sorted(dataset_path.glob("dataset-*of*.jsonl"))
    
    all_data = []
    
    if len(dataset_files) > 0:
        print(f"Loading {len(dataset_files)} partition files...")
        for file_path in tqdm(dataset_files, desc="Loading partitions"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            all_data.append(data)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse JSON in {file_path}: {e}")
                            continue
    else:
        # single partition file: prefer completions.jsonl (async inference
        # driver convention) over dataset.jsonl (legacy upstream convention).
        candidates = list(dataset_path.glob("completions.jsonl")) or list(dataset_path.glob("dataset.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No completions.jsonl or dataset.jsonl under {dataset_path}"
            )
        file_path = candidates[0]
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse JSON in {file_path}: {e}")
                        continue
    
    return all_data

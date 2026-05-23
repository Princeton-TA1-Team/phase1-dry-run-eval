"""
Visualization utilities for evaluation results.
"""

import os
import matplotlib

if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Any
import os
import numpy as np
from collections import defaultdict, Counter
import re
from .evaluation_utils import pass_and_maj_at_k_by_source


def extract_experiment_metadata(dataset: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
    """
    Extract experiment metadata from dataset and output path for figure headers.
    
    Args:
        dataset: The dataset with entries containing generation_metadata
        output_path: Path that may contain experiment information
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {
        'model_name': 'Unknown Model',
        'num_responses': 'Unknown',
        'total_questions': len(dataset),
        'sampling_params': {},
        'experiment_name': 'Unknown Experiment'
    }
    
    # Extract from first entry's generation_metadata if available
    if dataset and 'generation_metadata' in dataset[0]:
        gen_meta = dataset[0]['generation_metadata']
        
        # Extract model information
        if 'model_config_alias' in gen_meta:
            metadata['model_name'] = gen_meta['model_config_alias']
        
        # Extract number of responses
        if 'num_responses' in gen_meta:
            metadata['num_responses'] = gen_meta['num_responses']
        
        # Extract sampling parameters
        sampling_keys = ['use_math_prompt', 'enable_thinking', 'num_partitions']
        for key in sampling_keys:
            if key in gen_meta:
                metadata['sampling_params'][key] = gen_meta[key]
    
    # Extract experiment info from output path
    # Pattern: .../outputs/ExperimentType/ModelName/experiment_details/...
    path_parts = output_path.split('/')
    
    # Look for experiment information in path
    for i, part in enumerate(path_parts):
        if part == 'outputs' and i + 2 < len(path_parts):
            experiment_type = path_parts[i + 1]
            model_dir = path_parts[i + 2]
            if i + 3 < len(path_parts):
                experiment_details = path_parts[i + 3]
                metadata['experiment_name'] = f"{experiment_type}/{model_dir}"
                
                # Try to extract sampling parameters from experiment name
                # Pattern like: llama8b_SR_0.1_0.9_10k-n4-seed42
                if 'SR_' in experiment_details:
                    match = re.search(r'SR_([\d.]+)_([\d.]+)', experiment_details)
                    if match:
                        metadata['sampling_params']['min_p'] = float(match.group(1))
                        metadata['sampling_params']['max_p'] = float(match.group(2))
                
                if 'n' in experiment_details:
                    match = re.search(r'-n(\d+)', experiment_details)
                    if match:
                        metadata['sampling_params']['n_samples'] = int(match.group(1))
                
                if 'seed' in experiment_details:
                    match = re.search(r'seed(\d+)', experiment_details)
                    if match:
                        metadata['sampling_params']['seed'] = int(match.group(1))
                
                if 'k' in experiment_details:
                    match = re.search(r'(\d+)k', experiment_details)
                    if match:
                        metadata['sampling_params']['dataset_size'] = f"{match.group(1)}k"
            break
    
    return metadata


def format_metadata_header(metadata: Dict[str, Any]) -> str:
    """
    Format metadata into a readable header string for figures.
    
    Args:
        metadata: Dictionary with experiment metadata
        
    Returns:
        Formatted header string
    """
    lines = []
    
    # Model and experiment info
    lines.append(f"Model: {metadata['model_name']} | Experiment: {metadata['experiment_name']}")
    
    # Dataset info
    lines.append(f"Questions: {metadata['total_questions']:,} | Responses per question: {metadata['num_responses']}")
    
    # Sampling parameters
    if metadata['sampling_params']:
        param_strs = []
        for key, value in metadata['sampling_params'].items():
            if key == 'use_math_prompt':
                param_strs.append(f"Math prompt: {value}")
            elif key == 'enable_thinking':
                param_strs.append(f"Thinking: {value}")
            elif key == 'min_p' and 'max_p' in metadata['sampling_params']:
                param_strs.append(f"Sampling range: [{value}, {metadata['sampling_params']['max_p']}]")
            elif key == 'max_p':
                continue  # Already handled with min_p
            elif key == 'seed':
                param_strs.append(f"Seed: {value}")
            elif key == 'dataset_size':
                param_strs.append(f"Dataset: {value}")
            elif key == 'n_samples':
                param_strs.append(f"Trajectories: {value}")
            else:
                param_strs.append(f"{key}: {value}")
        
        if param_strs:
            lines.append(" | ".join(param_strs))
    
    return "\n".join(lines)


def create_unified_correctness_plot(dataset: List[Dict[str, Any]], output_path: str, answer_column: str, response_column: str):
    return

    # TODO: Fix this function
    """
    Create a unified plot showing correctness statistics including:
    1. Overall correctness rate and pass@k metrics by data source
    2. Distribution of correct answers per question for each data source
    """
    # Extract experiment metadata for header
    metadata = extract_experiment_metadata(dataset, output_path)
    header_text = format_metadata_header(metadata)
    
    # Calculate pass@k metrics using the utility function
    pass_at_k_stats = pass_and_maj_at_k_by_source(dataset, answer_column, response_column)
    
    if not pass_at_k_stats:
        print("No data sources found for visualization.")
        return
    
    sources = list(pass_at_k_stats.keys())
    n_sources = len(sources)
    
    # Create figure with subplots (add extra space for header)
    fig = plt.figure(figsize=(8, 4 * n_sources + 1))
    
    for i, source in enumerate(sources):
        stats = pass_at_k_stats[source]
        pass_at_k = stats['pass_at_k']
        total_questions = stats['total_questions']
        n_trajectories = stats['n_trajectories']
        overall_correctness = stats['overall_correctness']
        
        # Calculate correctness distribution for this source
        source_data = []
        for entry in dataset:
            if entry.get('source') == source and response_column in entry:
                question_responses = [response.get('correctness') is True for response in entry[response_column]]
                if question_responses:
                    correct_count = sum(question_responses)
                    fraction = correct_count / len(question_responses)
                    source_data.append(fraction)
        
        # Create subplots for this source (2 plots side by side)
        ax1 = plt.subplot(n_sources, 2, 2*i + 1)
        ax2 = plt.subplot(n_sources, 2, 2*i + 2)
        
        # Plot 1: Pass@k metrics and overall correctness
        k_values = list(range(1, n_trajectories + 1))
        
        # Bar plot for pass@k
        bars = ax1.bar(k_values, pass_at_k, alpha=0.7, color='skyblue', edgecolor='navy', label='Pass@k')
        
        # Add overall correctness rate as a horizontal line
        ax1.axhline(y=overall_correctness, color='red', linestyle='--', linewidth=2, 
                   label=f'Overall Rate: {overall_correctness:.1f}%')
        
        ax1.set_title(f'{source} - Pass@k Metrics\n({total_questions} questions, {n_trajectories} trajectories each)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('k (number of trajectories)', fontsize=10)
        ax1.set_ylabel('Pass@k Rate (%)', fontsize=10)
        ax1.set_ylim(0, 100)
        ax1.grid(axis='y', alpha=0.3)
        ax1.legend()
        
        # Add value labels on bars
        for bar, rate in zip(bars, pass_at_k):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Correctness distribution
        if source_data:
            bin_labels = [f'{i}/{n_trajectories}' for i in range(n_trajectories + 1)]
            
            # Count frequencies for each bin
            counts = []
            for i in range(n_trajectories + 1):
                target_fraction = i / n_trajectories
                count = sum(1 for frac in source_data if abs(frac - target_fraction) < 1e-10)
                counts.append(count)
            
            # Create bar plot
            x_pos = range(len(bin_labels))
            bars2 = ax2.bar(x_pos, counts, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars based on correctness level
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars2)))
            for bar, color in zip(bars2, colors):
                bar.set_color(color)
            
            ax2.set_title(f'{source} - Correctness Distribution\n(Questions by fraction correct)', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Fraction of Correct Answers per Question', fontsize=10)
            ax2.set_ylabel('Number of Questions', fontsize=10)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars2, counts):
                if count > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                           f'{count}', ha='center', va='bottom', fontsize=9)
            
            # Add mean correctness text
            mean_correct = np.mean(source_data)
            ax2.text(0.02, 0.98, f'Mean: {mean_correct:.3f}', transform=ax2.transAxes, 
                   va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, f'No data for {source}', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'{source} - No Data Available', fontsize=12, fontweight='bold')
    
    # Add experiment metadata header at the top of the figure
    fig.suptitle(header_text, fontsize=10, ha='center', va='top', y=0.98, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Leave space for header
    
    # Save the plot
    plot_path = output_path.replace('.jsonl', '_unified_correctness_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Unified correctness analysis plot saved to: {plot_path}")
    
    plt.close()


def create_correctness_distribution_plot(dataset: List[Dict[str, Any]], output_path: str):
    """
    Create a bar plot showing the distribution of correct answers per question for each data source.
    Shows how many questions have 0/n, 1/n, ..., n/n correct answers where n is the number of sampled trajectories.
    """
    # Group data by source
    source_data = defaultdict(list)
    
    for entry in dataset:
        source = entry.get('source', 'unknown')
        if response_column not in entry:
            continue
        
        # Count correct responses for this question
        total_responses = len(entry[response_column])
        correct_responses = sum(1 for response in entry[response_column] if response.get('correctness') is True)
        
        # Calculate the fraction of correct responses
        if total_responses > 0:
            correct_fraction = correct_responses / total_responses
            source_data[source].append(correct_fraction)
    
    if not source_data:
        print("No data sources found for correctness distribution visualization.")
        return
    
    # Create subplots for each source
    sources = list(source_data.keys())
    n_sources = len(sources)
    
    # Calculate layout
    if n_sources <= 2:
        rows, cols = 1, n_sources
        figsize = (8 * n_sources, 6)
    elif n_sources <= 4:
        rows, cols = 2, 2
        figsize = (12, 10)
    else:
        rows = int(np.ceil(n_sources / 3))
        cols = 3
        figsize = (15, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_sources == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Hide unused subplots
    for i in range(n_sources, len(axes)):
        axes[i].set_visible(False)
    
    # Create plots for each source
    for i, source in enumerate(sources):
        ax = axes[i]
        fractions = source_data[source]
        
        if not fractions:
            ax.text(0.5, 0.5, f'No data for {source}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{source} (0 questions)')
            continue
        
        # Find the number of trajectory samples (assuming it's consistent within a source)
        # We'll use the denominator from the fractions to determine possible values
        unique_fractions = list(set(fractions))
        
        # Determine the number of trajectories by finding the finest granularity
        n_trajectories = 1
        for frac in unique_fractions:
            if frac > 0 and frac < 1:
                # Find the smallest denominator that could create this fraction
                for denom in range(2, 21):  # Check up to 20 trajectories
                    if abs(frac - round(frac * denom) / denom) < 1e-10:
                        n_trajectories = max(n_trajectories, denom)
                        break
        
        # Create bins for 0/n, 1/n, 2/n, ..., n/n
        bins = [i / n_trajectories for i in range(n_trajectories + 1)]
        bin_labels = [f'{i}/{n_trajectories}' for i in range(n_trajectories + 1)]
        
        # Count frequencies for each bin
        counts = []
        for i in range(n_trajectories + 1):
            target_fraction = i / n_trajectories
            count = sum(1 for frac in fractions if abs(frac - target_fraction) < 1e-10)
            counts.append(count)
        
        # Create bar plot
        x_pos = range(len(bin_labels))
        bars = ax.bar(x_pos, counts, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Color bars based on correctness level
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Customize plot
        ax.set_xlabel('Fraction of Correct Answers per Question', fontsize=10)
        ax.set_ylabel('Number of Questions', fontsize=10)
        ax.set_title(f'{source}\n({len(fractions)} questions, {n_trajectories} trajectories each)', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
                       f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Add statistics text
        if fractions:
            mean_correct = np.mean(fractions)
            ax.text(0.02, 0.98, f'Mean: {mean_correct:.3f}', transform=ax.transAxes, 
                   va='top', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = output_path.replace('.jsonl', '_correctness_distribution.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Correctness distribution plot saved to: {plot_path}")
    
    plt.close()


def create_finish_reason_plot(error_stats: Dict[str, Any], output_path: str, dataset: List[Dict[str, Any]] = None):
    """
    Create a bar plot showing finish reason vs correctness statistics (no pie chart).
    """
    finish_reason_stats = error_stats.get('finish_reason_stats', {})
    finish_reason_correctness = error_stats.get('finish_reason_correctness', {})
    
    if not finish_reason_stats or not finish_reason_correctness:
        print("No finish reason data found for visualization.")
        return
    
    # Prepare data for plotting
    reasons = list(finish_reason_stats.keys())
    
    # Prepare data for stacked bar chart
    correct_counts = []
    incorrect_counts = []
    unparsable_counts = []
    
    for reason in reasons:
        stats = finish_reason_correctness.get(reason, {})
        correct_counts.append(stats.get('correct', 0))
        incorrect_counts.append(stats.get('incorrect', 0))
        unparsable_counts.append(stats.get('unparsable', 0))
    
    # Create the plot with extra space for header
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Add metadata header if dataset is provided
    if dataset:
        metadata = extract_experiment_metadata(dataset, output_path)
        header_text = format_metadata_header(metadata)
        fig.suptitle(header_text, fontsize=10, ha='center', va='top', y=0.95,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Create stacked bar chart
    x = range(len(reasons))
    ax.bar(x, correct_counts, label='Correct', color='green', alpha=0.7)
    ax.bar(x, incorrect_counts, bottom=correct_counts, label='Incorrect', color='red', alpha=0.7)
    ax.bar(x, unparsable_counts, bottom=[c + i for c, i in zip(correct_counts, incorrect_counts)], 
           label='Unparsable', color='gray', alpha=0.7)
    
    ax.set_title('Finish Reason vs Correctness', fontsize=14, fontweight='bold')
    ax.set_xlabel('Finish Reason', fontsize=12)
    ax.set_ylabel('Number of Responses', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(reasons, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    for i, reason in enumerate(reasons):
        total_count = correct_counts[i] + incorrect_counts[i] + unparsable_counts[i]
        if total_count > 0:
            correct_pct = correct_counts[i] / total_count * 100
            ax.text(i, total_count + max(correct_counts + incorrect_counts + unparsable_counts) * 0.01,
                   f'{correct_pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Adjust layout based on whether header is present
    if dataset:
        plt.tight_layout(rect=[0, 0, 1, 0.88])
    else:
        plt.tight_layout()
    
    # Save the plot
    plot_path = output_path.replace('.jsonl', '_finish_reasons.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Finish reason plot saved to: {plot_path}")
    
    plt.close()

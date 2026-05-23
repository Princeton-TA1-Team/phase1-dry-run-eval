"""
Output utilities for saving results and printing summaries.
"""

import json
from typing import Dict, List, Any


def print_evaluation_summary(dataset: List[Dict[str, Any]], error_stats: Dict[str, Any], response_column: str, verbose: bool = False):
    """
    Print summary statistics of the evaluation.
    """
    total_questions = len(dataset)
    total_responses = 0
    correct_responses = 0
    incorrect_responses = 0
    no_answer_responses = 0
    
    for entry in dataset:
        if response_column not in entry:
            continue
            
        for response in entry[response_column]:
            total_responses += 1
            correctness = response.get('correctness')
            
            if correctness is True:
                correct_responses += 1
            elif correctness is False:
                incorrect_responses += 1
            else:  # None
                no_answer_responses += 1
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total questions: {total_questions}")
    print(f"Total responses: {total_responses}")
    print(f"Correct responses: {correct_responses} ({correct_responses/total_responses*100:.2f}%)")
    print(f"Incorrect responses: {incorrect_responses} ({incorrect_responses/total_responses*100:.2f}%)")
    print(f"Unparsable responses: {no_answer_responses} ({no_answer_responses/total_responses*100:.2f}%)")
    print("="*60)
    
    # Detailed error analysis
    print("\nDETAILED ERROR ANALYSIS")
    print("-" * 40)
    print(f"Questions with all incorrect responses: {error_stats['questions_with_all_incorrect']} ({error_stats['questions_with_all_incorrect']/total_questions*100:.2f}%)")
    print(f"Questions with all unparsable responses: {error_stats['questions_with_all_unparsable']} ({error_stats['questions_with_all_unparsable']/total_questions*100:.2f}%)")
    
    # Show finish reason statistics
    print(f"\nFINISH REASON STATISTICS:")
    for reason, count in sorted(error_stats['finish_reason_stats'].items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_responses * 100
        print(f"  {reason}: {count} ({percentage:.2f}%)")
    
    # Show finish reason vs correctness correlation
    print(f"\nFINISH REASON vs CORRECTNESS ANALYSIS:")
    for reason, stats in error_stats['finish_reason_correctness'].items():
        total_for_reason = sum(stats.values())
        if total_for_reason > 0:
            correct_pct = stats['correct'] / total_for_reason * 100
            incorrect_pct = stats['incorrect'] / total_for_reason * 100
            unparsable_pct = stats['unparsable'] / total_for_reason * 100
            print(f"  {reason} (n={total_for_reason}):")
            print(f"    Correct: {stats['correct']} ({correct_pct:.1f}%)")
            print(f"    Incorrect: {stats['incorrect']} ({incorrect_pct:.1f}%)")
            print(f"    Unparsable: {stats['unparsable']} ({unparsable_pct:.1f}%)")
    
    if not verbose:
        print("="*60)
        return
    
    # Show top incorrect patterns
    if error_stats['common_incorrect_patterns']:
        print(f"\nTop 5 most common incorrect answer patterns:")
        sorted_patterns = sorted(error_stats['common_incorrect_patterns'].items(), key=lambda x: x[1], reverse=True)
        for i, (pattern, count) in enumerate(sorted_patterns[:5]):
            print(f"  {i+1}. '{pattern}...' - {count} occurrences")
    
    # Show sample unparsable responses
    if error_stats['unparsable_responses']:
        print(f"\nSample unparsable responses (first 3):")
        for i, unparsable in enumerate(error_stats['unparsable_responses'][:3]):
            print(f"  {i+1}. Question ID: {unparsable['question_id']}")
            print(f"     Ground truth: {unparsable['ground_truth']}")
            print(f"     Response preview: {unparsable['response_text']}...")
            print()
    
    print("="*60)


def save_evaluated_dataset(dataset: List[Dict[str, Any]], output_path: str, flatten: bool = False, response_column: str = 'responses'):
    """
    Save the evaluated dataset to a JSONL file.
    """
    print(f"Saving evaluated dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"Dataset saved successfully!")

    if flatten:
        flattened_path = output_path.replace('.jsonl', '_flattened.jsonl')
        print(f"Saving flattened dataset to {flattened_path}...")
        with open(flattened_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                responses = entry.pop(response_column, [])
                for response in responses:
                    flat_response = {f"{response_column}_{k}": v for k, v in response.items()}
                    flat_entry = {**entry, **flat_response}
                    f.write(json.dumps(flat_entry, ensure_ascii=False) + '\n')
        print(f"Flattened dataset saved successfully!")


def save_error_analysis(error_stats: Dict[str, Any], output_path: str):
    """
    Save detailed error analysis to a JSON file.
    """
    error_analysis_path = output_path.replace('.jsonl', '_error_analysis.json')
    print(f"Saving error analysis to {error_analysis_path}...")
    
    # Convert to serializable format
    serializable_stats = {
        'pass_at_k_by_source': error_stats.get('pass_at_k_by_source', {}),
        'overall_stats': error_stats.get('overall_stats', {}),
        'sample_incorrect_answers': error_stats.get('sample_incorrect_answers', []),
        'unparsable_responses_count': len(error_stats.get('unparsable_responses', [])),
        'questions_with_all_incorrect': error_stats.get('questions_with_all_incorrect', 0),
        'questions_with_all_unparsable': error_stats.get('questions_with_all_unparsable', 0),
        'finish_reason_stats': error_stats.get('finish_reason_stats', {}),
        'finish_reason_correctness': error_stats.get('finish_reason_correctness', {}),
        'common_incorrect_patterns': error_stats.get('common_incorrect_patterns', {}),
        'sample_unparsable_responses': error_stats.get('unparsable_responses', [])[:10]  # First 10
    }
    
    with open(error_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    print(f"Error analysis saved successfully!")

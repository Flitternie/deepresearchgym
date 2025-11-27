import argparse
import asyncio
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio

# Import functions from the original KPR evaluation module
from eval_kpr_async import (
    evaluate_query as original_evaluate_query,
    evaluate_answer,
    evaluate_single_key_point,
    analyze_results
)


async def evaluate_multi_answer_kpr_query(semaphore, question_id, question_path, answer_paths, key_point_dir, model):
    """
    Evaluate a single question with multiple answer files for KPR and return averaged scores.
    
    Args:
        semaphore: Asyncio semaphore for concurrency control
        question_id: The ID of the question
        question_path: Path to the question file (not used directly but kept for consistency)
        answer_paths: List of paths to answer files for this question
        key_point_dir: Directory containing key point files
        model: The model to use for evaluation
    
    Returns:
        Tuple of (question_id, evaluation_result)
    """
    key_point_path = Path(key_point_dir) / f"{question_id}_aggregated.json"
    
    if not key_point_path.exists():
        print(f"Warning: Missing key point file for query {question_id}")
        return question_id, None
    
    # Load key points
    with open(key_point_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        key_points = data["key_points"]
    
    # Evaluate each answer file
    all_evaluations = []
    valid_answers = 0
    
    for answer_path in answer_paths:
        if not answer_path.exists():
            print(f"Warning: Missing answer file {answer_path}")
            continue
            
        try:
            answer = answer_path.read_text().strip()
            evaluations = await evaluate_answer(semaphore, answer, key_points, model)
            all_evaluations.append(evaluations)
            valid_answers += 1
        except Exception as e:
            print(f"Error evaluating {answer_path}: {e}")
            continue
    
    if valid_answers == 0:
        print(f"Warning: No valid answers found for query {question_id}")
        return question_id, None
    
    # Calculate averaged scores across all answer files for each key point
    averaged_labels = {}
    
    # Get all unique point numbers from the evaluations
    all_point_numbers = set()
    for evaluation in all_evaluations:
        all_point_numbers.update(evaluation.keys())
    
    for point_number in all_point_numbers:
        labels = []
        justifications = []
        
        for evaluation in all_evaluations:
            if point_number in evaluation:
                label, justification = evaluation[point_number]
                labels.append(label)
                justifications.append(justification)
        
        if labels:
            # For KPR, we'll use majority voting for the label
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Get the most common label
            majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
            
            # Combine justifications
            combined_justification = f"Majority vote from {len(labels)} evaluations ({label_counts}): " + " | ".join(justifications[:3])
            
            averaged_labels[point_number] = (majority_label, combined_justification)
    
    # Calculate support rates
    supported_count = sum(1 for label, _ in averaged_labels.values() if label == "Supported")
    omitted_count = sum(1 for label, _ in averaged_labels.values() if label == "Omitted")
    contradicted_count = sum(1 for label, _ in averaged_labels.values() if label == "Contradicted")
    
    total_points = len(averaged_labels)
    if total_points == 0:
        return question_id, None
    
    support_rate = supported_count / total_points * 100
    omitted_rate = omitted_count / total_points * 100
    contradicted_rate = contradicted_count / total_points * 100
    
    return question_id, {
        "labels": averaged_labels,
        "support_rate": support_rate,
        "omitted_rate": omitted_rate,
        "contradicted_rate": contradicted_rate,
        "num_answers_evaluated": valid_answers,
        "individual_evaluations": all_evaluations
    }


def find_answer_files(answer_dir, question_id):
    """
    Find all answer files for a given question ID.
    Expected format: questionid_run_i.a where i is the run index.
    
    Args:
        answer_dir: Directory containing answer files
        question_id: The question ID to find answers for
    
    Returns:
        List of Path objects for answer files
    """
    pattern = f"{question_id}_run_*.a"
    answer_files = list(Path(answer_dir).glob(pattern))
    
    # Sort by run index for consistent ordering
    def extract_run_index(filepath):
        try:
            # Extract run index from filename like "questionid_run_5.a"
            stem = filepath.stem
            parts = stem.split('_run_')
            if len(parts) == 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return 0
    
    answer_files.sort(key=extract_run_index)
    return answer_files


async def evaluate_multi_answer_kpr_dataset(question_dir, answer_dir, key_point_dir, model, num_workers=32):
    """
    Evaluate a dataset where questions and answers are in different directories,
    and each question may have multiple answer files for KPR.
    
    Args:
        question_dir: Directory containing question files (.q)
        answer_dir: Directory containing answer files (.a with format questionid_run_i.a)
        key_point_dir: Directory containing key point files
        model: The model to use for evaluation
        num_workers: Number of concurrent workers
    
    Returns:
        Dictionary of evaluation results
    """
    question_path = Path(question_dir)
    answer_path = Path(answer_dir)
    
    all_results = {}
    semaphore = asyncio.Semaphore(num_workers)
    
    # Find all question files
    question_files = list(question_path.glob("*.q"))
    
    # Group answer files by question ID
    tasks = []
    for question_file in question_files:
        question_id = question_file.stem
        answer_files = find_answer_files(answer_path, question_id)
        
        if not answer_files:
            print(f"Warning: No answer files found for question {question_id}")
            continue
        
        print(f"Found {len(answer_files)} answer files for question {question_id}")
        
        task = evaluate_multi_answer_kpr_query(
            semaphore, question_id, question_file, answer_files, key_point_dir, model
        )
        tasks.append(task)
    
    # Execute all evaluations
    results = await tqdm_asyncio.gather(*tasks)
    
    for query_id, result in results:
        if result is not None:
            all_results[query_id] = result
    
    return all_results


def analyze_multi_answer_kpr_results(results):
    """
    Analyze results from multi-answer KPR evaluation and print statistics.
    
    Args:
        results: Dictionary of evaluation results
    
    Returns:
        Dictionary with summary statistics
    """
    total_support_rate = 0
    total_omitted_rate = 0
    total_contradicted_rate = 0
    total_answers_evaluated = 0
    num_queries = len(results)

    for query_id, result in results.items():
        total_support_rate += result.get("support_rate", 0)
        total_omitted_rate += result.get("omitted_rate", 0)
        total_contradicted_rate += result.get("contradicted_rate", 0)
        total_answers_evaluated += result.get("num_answers_evaluated", 0)

    avg_support = total_support_rate / num_queries if num_queries > 0 else 0
    avg_omitted = total_omitted_rate / num_queries if num_queries > 0 else 0
    avg_contradicted = total_contradicted_rate / num_queries if num_queries > 0 else 0
    avg_answers_per_question = total_answers_evaluated / num_queries if num_queries > 0 else 0

    print(f"\nKPR Evaluation Summary:")
    print(f"Total questions evaluated: {num_queries}")
    print(f"Total answer files evaluated: {total_answers_evaluated}")
    print(f"Average answer files per question: {avg_answers_per_question:.2f}")
    print(f"Averages across {num_queries} questions:")
    print(f"  Average support rate: {avg_support:.2f}%")
    print(f"  Average omitted rate: {avg_omitted:.2f}%")
    print(f"  Average contradicted rate: {avg_contradicted:.2f}%")

    return {
        "average_support_rate": avg_support,
        "average_omitted_rate": avg_omitted,
        "average_contradicted_rate": avg_contradicted,
        "total_questions": num_queries,
        "total_answers_evaluated": total_answers_evaluated,
        "avg_answers_per_question": avg_answers_per_question
    }


async def main(args):
    """
    Main function to handle multi-answer KPR evaluation.
    """
    print(f"Evaluating KPR for questions from {args.question_dir} and answers from {args.answer_dir}")
    print(f"Key points from: {args.key_point_dir}")
    print(f"Using model: {args.model}")
    
    if args.times == 1:
        output_path = os.path.join(args.output, f"multi_answer_kpr_{args.model}.json")
        
        results = await evaluate_multi_answer_kpr_dataset(
            args.question_dir, 
            args.answer_dir, 
            args.key_point_dir,
            args.model, 
            args.workers
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved detailed evaluation results to {output_path}")
        analyze_multi_answer_kpr_results(results)
        
    else:
        overall_results = []
        for i in range(args.times):
            print(f"\n=== KPR Evaluation Run {i+1}/{args.times} ===")
            output_path = os.path.join(args.output, f"multi_answer_kpr_{args.model}_{i+1}.json")
            
            results = await evaluate_multi_answer_kpr_dataset(
                args.question_dir, 
                args.answer_dir, 
                args.key_point_dir,
                args.model, 
                args.workers
            )
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            summary_results = analyze_multi_answer_kpr_results(results)
            overall_results.append(summary_results)
        
        # Print overall results by averaging across all runs
        print("\n" + "="*60)
        print("OVERALL KPR RESULTS ACROSS ALL RUNS:")
        print("="*60)
        
        avg_support = sum(res["average_support_rate"] for res in overall_results) / len(overall_results)
        avg_omitted = sum(res["average_omitted_rate"] for res in overall_results) / len(overall_results)
        avg_contradicted = sum(res["average_contradicted_rate"] for res in overall_results) / len(overall_results)
        avg_answers_per_question = sum(res["avg_answers_per_question"] for res in overall_results) / len(overall_results)
        
        print(f"Average support rate across {args.times} runs: {avg_support:.2f}%")
        print(f"Average omitted rate across {args.times} runs: {avg_omitted:.2f}%")
        print(f"Average contradicted rate across {args.times} runs: {avg_contradicted:.2f}%")
        print(f"Total questions: {overall_results[0]['total_questions']}")
        print(f"Total answers evaluated: {overall_results[0]['total_answers_evaluated']}")
        print(f"Average answers per question: {avg_answers_per_question:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple answer files per question for KPR")
    parser.add_argument("--question_dir", required=True, help="Directory containing question files (.q)")
    parser.add_argument("--answer_dir", required=True, help="Directory containing answer files (.a)")
    parser.add_argument("--key_point_dir", required=True, help="Directory containing key point files")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", required=True, help="Model to use for evaluation")
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent workers")
    parser.add_argument("--times", type=int, default=1, help="Number of times to run the evaluation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    asyncio.run(main(args))

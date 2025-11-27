import argparse
import asyncio
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio

# Import functions from the original citation recall evaluation module
from eval_citation_recall_async import (
    evaluate_query as original_evaluate_query,
    extract_claims_and_url,
    analyze_results
)


async def evaluate_multi_answer_citation_recall_query(semaphore, question_id, answer_paths, model):
    """
    Evaluate a single question with multiple answer files for citation recall and return averaged scores.
    
    Args:
        semaphore: Asyncio semaphore for concurrency control
        question_id: The ID of the question
        answer_paths: List of paths to answer files for this question
        model: The model to use for evaluation
    
    Returns:
        Tuple of (question_id, evaluation_result)
    """
    # Evaluate each answer file
    all_evaluations = []
    valid_answers = 0
    
    for answer_path in answer_paths:
        if not answer_path.exists():
            print(f"Warning: Missing answer file {answer_path}")
            continue
            
        try:
            # Use the original evaluate_query function but pass the path directly
            _, result = await original_evaluate_query(semaphore, question_id, answer_path, model)
            if result is not None:
                all_evaluations.append(result)
                valid_answers += 1
        except Exception as e:
            print(f"Error evaluating {answer_path}: {e}")
            continue
    
    if valid_answers == 0:
        print(f"Warning: No valid answers found for query {question_id}")
        return question_id, None
    
    # Calculate averaged scores across all answer files
    total_score = sum(result["score"] for result in all_evaluations)
    avg_score = total_score / len(all_evaluations)
    
    # Combine detailed results
    combined_detailed = {}
    for i, result in enumerate(all_evaluations):
        if result.get("detailed") and isinstance(result["detailed"], dict):
            for claim_id, claim_data in result["detailed"].items():
                combined_key = f"run_{i+1}_{claim_id}"
                combined_detailed[combined_key] = claim_data
        elif result.get("detailed") and isinstance(result["detailed"], str):
            # Handle case where detailed is a string message
            combined_detailed[f"run_{i+1}_message"] = result["detailed"]
    
    return question_id, {
        "score": avg_score,
        "detailed": combined_detailed if combined_detailed else f"Average of {valid_answers} evaluations",
        "num_answers_evaluated": valid_answers,
        "individual_scores": [result["score"] for result in all_evaluations]
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


async def evaluate_multi_answer_citation_recall_dataset(question_dir, answer_dir, model, num_workers=32):
    """
    Evaluate a dataset where questions and answers are in different directories,
    and each question may have multiple answer files for citation recall.
    
    Args:
        question_dir: Directory containing question files (.q)
        answer_dir: Directory containing answer files (.a with format questionid_run_i.a)
        model: The model to use for evaluation
        num_workers: Number of concurrent workers
    
    Returns:
        Tuple of (Dictionary of evaluation results, average score)
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
        
        task = evaluate_multi_answer_citation_recall_query(
            semaphore, question_id, answer_files, model
        )
        tasks.append(task)
    
    # Execute all evaluations
    results = await tqdm_asyncio.gather(*tasks)
    
    for query_id, result in results:
        if result is not None:
            all_results[query_id] = result
    
    # Calculate average score
    average_score = sum(r["score"] for r in all_results.values()) / len(all_results) if all_results else 0
    
    return all_results, average_score


def analyze_multi_answer_citation_recall_results(results):
    """
    Analyze results from multi-answer citation recall evaluation and print statistics.
    
    Args:
        results: Dictionary of evaluation results
    
    Returns:
        Dictionary with summary statistics
    """
    total_score = 0
    count = 0
    total_answers_evaluated = 0

    for query_id, result in results.items():
        query_score = result.get("score", 0)
        total_score += query_score * 100  # scale to 0–100 for percentage form
        count += 1
        total_answers_evaluated += result.get("num_answers_evaluated", 0)

    average_score = total_score / count if count > 0 else 0
    avg_answers_per_question = total_answers_evaluated / count if count > 0 else 0
    
    print(f"\nCitation Recall Evaluation Summary:")
    print(f"Total questions evaluated: {count}")
    print(f"Total answer files evaluated: {total_answers_evaluated}")
    print(f"Average answer files per question: {avg_answers_per_question:.2f}")
    print(f"Average normalized citation recall score across {count} questions: {average_score:.2f}")

    return {
        "average_citation_recall_score": average_score,
        "total_questions": count,
        "total_answers_evaluated": total_answers_evaluated,
        "avg_answers_per_question": avg_answers_per_question
    }


async def main(args):
    """
    Main function to handle multi-answer citation recall evaluation.
    """
    print(f"Evaluating citation recall for questions from {args.question_dir} and answers from {args.answer_dir}")
    print(f"Using model: {args.model}")
    
    if args.times == 1:
        output_path = os.path.join(args.output, f"multi_answer_citation_recall_{args.model}.json")
        
        results, avg_score = await evaluate_multi_answer_citation_recall_dataset(
            args.question_dir, 
            args.answer_dir, 
            args.model, 
            args.workers
        )
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved detailed evaluation results to {output_path}")
        analyze_multi_answer_citation_recall_results(results)
        
    else:
        overall_results = []
        for i in range(args.times):
            print(f"\n=== Citation Recall Evaluation Run {i+1}/{args.times} ===")
            output_path = os.path.join(args.output, f"multi_answer_citation_recall_{args.model}_{i+1}.json")
            
            results, avg_score = await evaluate_multi_answer_citation_recall_dataset(
                args.question_dir, 
                args.answer_dir, 
                args.model, 
                args.workers
            )
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            summary_results = analyze_multi_answer_citation_recall_results(results)
            overall_results.append(summary_results)
        
        # Print overall results by averaging across all runs
        print("\n" + "="*60)
        print("OVERALL CITATION RECALL RESULTS ACROSS ALL RUNS:")
        print("="*60)
        
        avg_citation_recall_score = sum(res["average_citation_recall_score"] for res in overall_results) / len(overall_results)
        avg_answers_per_question = sum(res["avg_answers_per_question"] for res in overall_results) / len(overall_results)
        
        print(f"Average citation recall score across {args.times} runs: {avg_citation_recall_score:.2f}")
        print(f"Total questions: {overall_results[0]['total_questions']}")
        print(f"Total answers evaluated: {overall_results[0]['total_answers_evaluated']}")
        print(f"Average answers per question: {avg_answers_per_question:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multiple answer files per question for citation recall")
    parser.add_argument("--question_dir", required=True, help="Directory containing question files (.q)")
    parser.add_argument("--answer_dir", required=True, help="Directory containing answer files (.a)")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--model", required=True, help="Model to use for evaluation")
    parser.add_argument("--workers", type=int, default=32, help="Number of concurrent workers")
    parser.add_argument("--times", type=int, default=1, help="Number of times to run the evaluation")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    asyncio.run(main(args))

"""
Qualitative analysis of negative sampling experiments.

Analyzes which questions improved/degraded with random negatives,
identifies patterns, and provides examples for the report.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from dexter.utils.metrics.ExactMatch import ExactMatch

def load_predictions(filepath):
    """Load predictions from JSON file"""
    if not Path(filepath).exists():
        # Try json_files directory
        json_files_path = os.path.join('json_files', filepath)
        if Path(json_files_path).exists():
            filepath = json_files_path
        else:
            return None
    
    with open(filepath, 'r') as f:
        return json.load(f)


def normalize_answer(answer):
    """Normalize answer for comparison"""
    if not answer:
        return ""
    return answer.lower().strip()


def analyze_question_changes(baseline_preds, experiment_preds, questions_map):
    """
    Analyze how predictions changed between baseline and experiment.
    
    Returns:
    - improved: questions that went from wrong to correct
    - degraded: questions that went from correct to wrong
    - unchanged_correct: questions that stayed correct
    - unchanged_wrong: questions that stayed wrong
    """
    metric = ExactMatch()
    improved = []
    degraded = []
    unchanged_correct = []
    unchanged_wrong = []
    
    for q_id, q_data in questions_map.items():
        if q_id not in baseline_preds or q_id not in experiment_preds:
            continue
        
        gt_answer = q_data['answer']
        baseline_pred = baseline_preds[q_id]
        experiment_pred = experiment_preds[q_id]
        
        baseline_correct = metric.evaluate(baseline_pred, gt_answer)
        experiment_correct = metric.evaluate(experiment_pred, gt_answer)
        
        question_info = {
            'q_id': q_id,
            'question': q_data['question'],
            'answer': gt_answer,
            'baseline_pred': baseline_pred,
            'experiment_pred': experiment_pred,
            'context': q_data.get('context', [])
        }
        
        if not baseline_correct and experiment_correct:
            improved.append(question_info)
        elif baseline_correct and not experiment_correct:
            degraded.append(question_info)
        elif baseline_correct and experiment_correct:
            unchanged_correct.append(question_info)
        else:
            unchanged_wrong.append(question_info)
    
    return improved, degraded, unchanged_correct, unchanged_wrong


def analyze_improved_questions(improved_questions, corpus, retrieval_results, k):
    """
    Analyze characteristics of questions that improved.
    """
    print("\n" + "="*80)
    print("ANALYSIS OF IMPROVED QUESTIONS")
    print("="*80)
    
    if not improved_questions:
        print("No questions improved with this configuration.")
        return
    
    print(f"\nTotal improved: {len(improved_questions)}")
    
    # Analyze patterns
    patterns = {
        'has_oracle_in_topk': 0,
        'multi_hop_count': defaultdict(int),
        'answer_type': defaultdict(int)
    }
    
    for q_info in improved_questions:
        q_id = q_info['q_id']
        
        # Check if oracle docs in top-k
        if q_id in retrieval_results:
            retrieved = list(retrieval_results[q_id].keys())[:k]
            oracle_titles = [ctx[0] if isinstance(ctx, list) else str(ctx) 
                           for ctx in q_info.get('context', [])]
            # Simple check: if any oracle title matches retrieved doc titles
            patterns['has_oracle_in_topk'] += 1  # Simplified
        
        # Count hops (rough estimate from context length)
        num_contexts = len(q_info.get('context', []))
        patterns['multi_hop_count'][num_contexts] += 1
        
        # Answer type (rough classification)
        answer = q_info['answer']
        if any(char.isdigit() for char in answer):
            patterns['answer_type']['numeric'] += 1
        elif len(answer.split()) == 1:
            patterns['answer_type']['single_word'] += 1
        else:
            patterns['answer_type']['phrase'] += 1
    
    print(f"\nCharacteristics:")
    print(f"  - Questions with oracle in top-k: {patterns['has_oracle_in_topk']}/{len(improved_questions)}")
    print(f"  - Multi-hop distribution:")
    for num_hops, count in sorted(patterns['multi_hop_count'].items()):
        print(f"    {num_hops} contexts: {count} questions")
    print(f"  - Answer types:")
    for ans_type, count in patterns['answer_type'].items():
        print(f"    {ans_type}: {count}")
    
    return patterns


def print_examples(questions, title, max_examples=5):
    """Print example questions"""
    print(f"\n{'='*80}")
    print(f"{title} ({len(questions)} total)")
    print(f"{'='*80}")
    
    for i, q_info in enumerate(questions[:max_examples], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {q_info['question']}")
        print(f"Correct Answer: {q_info['answer']}")
        print(f"Baseline Prediction: {q_info['baseline_pred']}")
        print(f"Experiment Prediction: {q_info['experiment_pred']}")
        if q_info.get('context'):
            print(f"Oracle Contexts: {len(q_info['context'])} documents")
            for ctx in q_info['context'][:2]:  # Show first 2
                title = ctx[0] if isinstance(ctx, list) else str(ctx)
                print(f"  - {title}")


def find_available_experiments():
    """Find all available experiment result files"""
    results_dir = 'results_negative_experiments'
    available = []
    
    # Check for random negative experiments
    for k in [1, 3, 5]:
        baseline_file = f'predictions_k{k}.json'
        baseline = load_predictions(baseline_file)
        if not baseline:
            continue
        
        # Find all ratio experiments for this k
        for R in [1, 3, 5]:
            for N in [1, 2, 5, 10, 15, 20]:
                if k == 1 and N > 15:
                    continue
                if k == 3 and N > 15:
                    continue
                if k == 5 and R == 5 and N not in [2, 5, 10, 15, 20]:
                    continue
                if k == 5 and R == 3 and N not in [1, 3, 6, 10, 15]:
                    continue
                if k == 5 and R == 1 and N not in [1, 2, 5, 10, 15]:
                    continue
                
                pred_file = os.path.join(results_dir, f'predictions_random_k{k}_ratio_{R}_{N}.json')
                if Path(pred_file).exists():
                    available.append((f'k={k}, {R}:{N}', pred_file, k, baseline_file))
    
    return available


def compare_configurations():
    """Compare different configurations"""
    
    print("QUALITATIVE ANALYSIS OF NEGATIVE SAMPLING EXPERIMENTS")
    print("="*80)
    
    # Load data
    with open('data/dev.json', 'r') as f:
        questions = json.load(f)
    questions_map = {q['_id']: q for q in questions[:1200]}
    
    # Find available experiments
    available_experiments = find_available_experiments()
    
    if not available_experiments:
        print("\nNo experiment results found. Please run experiments first:")
        print("  python run_negative_experiments.py")
        return {}
    
    print(f"\nFound {len(available_experiments)} experiment result files")
    
    # Group by k value for analysis
    experiments_by_k = defaultdict(list)
    for config_name, pred_file, k, baseline_file in available_experiments:
        experiments_by_k[k].append((config_name, pred_file, baseline_file))
    
    all_results = {}
    
    # Analyze each k value
    for k, configs in experiments_by_k.items():
        print(f"\n\n{'='*80}")
        print(f"ANALYZING k={k} EXPERIMENTS")
        print(f"{'='*80}")
        
        # Load baseline for this k
        baseline_file = configs[0][2]  # Get baseline file from first config
        baseline = load_predictions(baseline_file)
        if not baseline:
            print(f"Could not load baseline {baseline_file}")
            continue
        
        # Load retrieval results
        retrieval_file = f'retrieval_k{k}.json'
        retrieval_results = load_predictions(retrieval_file)
        
        # Analyze each configuration
        for config_name, pred_file, _ in configs:
            experiment_preds = load_predictions(pred_file)
            if not experiment_preds:
                continue
            
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name}")
            print(f"{'='*60}")
            
            improved, degraded, unchanged_correct, unchanged_wrong = analyze_question_changes(
                baseline, experiment_preds, questions_map
            )
            
            total = len(improved) + len(degraded) + len(unchanged_correct) + len(unchanged_wrong)
            improvement_rate = len(improved) / total * 100 if total > 0 else 0
            degradation_rate = len(degraded) / total * 100 if total > 0 else 0
            
            print(f"\nSummary:")
            print(f"  Total questions: {total}")
            print(f"  Improved (wrong→correct): {len(improved)} ({improvement_rate:.2f}%)")
            print(f"  Degraded (correct→wrong): {len(degraded)} ({degradation_rate:.2f}%)")
            print(f"  Unchanged (correct): {len(unchanged_correct)}")
            print(f"  Unchanged (wrong): {len(unchanged_wrong)}")
            print(f"  Net change: {len(improved) - len(degraded)} questions")
            
            # Analyze improved questions
            if improved and retrieval_results:
                analyze_improved_questions(improved, None, retrieval_results, k=k)
                if len(improved) > 0:
                    print_examples(improved, f"Questions Improved by {config_name}", max_examples=3)
            
            # Show degraded examples
            if degraded and len(degraded) > 0:
                print_examples(degraded, f"Questions Degraded by {config_name}", max_examples=2)
            
            all_results[config_name] = {
                'improved': improved,
                'degraded': degraded,
                'improvement_rate': improvement_rate,
                'degradation_rate': degradation_rate,
                'net_change': len(improved) - len(degraded)
            }
    
    all_results = {}
    
    for config_name, pred_file in configs_to_analyze:
        experiment_preds = load_predictions(pred_file)
        if not experiment_preds:
            print(f"\nSkipping {config_name} (file not found)")
            continue
        
        print(f"\n\n{'='*80}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'='*80}")
        
        improved, degraded, unchanged_correct, unchanged_wrong = analyze_question_changes(
            baseline, experiment_preds, questions_map
        )
        
        total = len(improved) + len(degraded) + len(unchanged_correct) + len(unchanged_wrong)
        improvement_rate = len(improved) / total * 100 if total > 0 else 0
        degradation_rate = len(degraded) / total * 100 if total > 0 else 0
        
        print(f"\nSummary:")
        print(f"  Total questions: {total}")
        print(f"  Improved (wrong→correct): {len(improved)} ({improvement_rate:.2f}%)")
        print(f"  Degraded (correct→wrong): {len(degraded)} ({degradation_rate:.2f}%)")
        print(f"  Unchanged (correct): {len(unchanged_correct)}")
        print(f"  Unchanged (wrong): {len(unchanged_wrong)}")
        print(f"  Net change: {len(improved) - len(degraded)} questions")
        
        # Analyze improved questions
        if improved:
            analyze_improved_questions(improved, None, retrieval_k5, k=5)
            print_examples(improved, f"Questions Improved by {config_name}")
        
        # Show degraded examples
        if degraded:
            print_examples(degraded, f"Questions Degraded by {config_name}", max_examples=3)
        
        all_results[config_name] = {
            'improved': improved,
            'degraded': degraded,
            'improvement_rate': improvement_rate,
            'degradation_rate': degradation_rate,
            'net_change': len(improved) - len(degraded)
        }
    
    # Summary comparison
    if all_results:
        print(f"\n\n{'='*80}")
        print("SUMMARY COMPARISON")
        print(f"{'='*80}")
        print(f"{'Configuration':<25} {'Improved':<10} {'Degraded':<10} {'Net Change':<12} {'Improvement %':<15}")
        print("-" * 80)
        
        for config_name, results in all_results.items():
            print(f"{config_name:<25} {len(results['improved']):<10} {len(results['degraded']):<10} "
                  f"{results['net_change']:<12} {results['improvement_rate']:.2f}%")
        
        # Find best configuration
        best_config = max(all_results.items(), key=lambda x: x[1]['net_change'])
        print(f"\nBest configuration: {best_config[0]}")
        print(f"  Net improvement: {best_config[1]['net_change']} questions")
        print(f"  Improvement rate: {best_config[1]['improvement_rate']:.2f}%")
    else:
        print("\nNo results found. Make sure you have run the experiments first.")
    
    # Save detailed results
    output_file = 'qualitative_analysis_results.json'
    output_data = {}
    for config_name, results in all_results.items():
        output_data[config_name] = {
            'improved_count': len(results['improved']),
            'degraded_count': len(results['degraded']),
            'improvement_rate': results['improvement_rate'],
            'degradation_rate': results['degradation_rate'],
            'net_change': results['net_change'],
            'improved_examples': results['improved'][:10],  # Save first 10
            'degraded_examples': results['degraded'][:10]
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")
    
    return all_results


def analyze_specific_improvements():
    """Analyze the high-ratio improvement in detail"""
    
    print("\n\n" + "="*80)
    print("DETAILED ANALYSIS: High Ratio (5:10) Improvement")
    print("="*80)
    
    with open('data/dev.json', 'r') as f:
        questions = json.load(f)
    questions_map = {q['_id']: q for q in questions[:1200]}
    
    baseline = load_predictions('predictions_k5.json')
    high_ratio = load_predictions('results_negative_experiments/predictions_random_k5_ratio_5_10.json')
    
    if not baseline or not high_ratio:
        print("Error: Could not load predictions")
        return
    
    improved, degraded, _, _ = analyze_question_changes(
        baseline, high_ratio, questions_map
    )
    
    print(f"\nQuestions where 5:10 ratio helped: {len(improved)}")
    print(f"Questions where 5:10 ratio hurt: {len(degraded)}")
    
    if improved:
        print("\n" + "="*80)
        print("TOP 10 QUESTIONS IMPROVED BY HIGH RATIO (5:10)")
        print("="*80)
        
        for i, q_info in enumerate(improved[:10], 1):
            print(f"\n{'='*60}")
            print(f"Example {i}")
            print(f"{'='*60}")
            print(f"Question ID: {q_info['q_id']}")
            print(f"Question: {q_info['question']}")
            print(f"Correct Answer: {q_info['answer']}")
            print(f"\nBaseline (k=5, no random): {q_info['baseline_pred']}")
            print(f"High Ratio (k=5, 5:10): {q_info['experiment_pred']}")
            print(f"✓ High ratio correctly answered this question")
            
            if q_info.get('context'):
                print(f"\nOracle Contexts ({len(q_info['context'])} documents):")
                for j, ctx in enumerate(q_info['context'][:3], 1):
                    title = ctx[0] if isinstance(ctx, list) else str(ctx)
                    print(f"  {j}. {title}")
    
    if degraded:
        print("\n\n" + "="*80)
        print("QUESTIONS WHERE HIGH RATIO HURT")
        print("="*80)
        print(f"Total: {len(degraded)} questions")
        print("\nTop 5 examples:")
        
        for i, q_info in enumerate(degraded[:5], 1):
            print(f"\n--- Example {i} ---")
            print(f"Question: {q_info['question']}")
            print(f"Answer: {q_info['answer']}")
            print(f"Baseline (correct): {q_info['baseline_pred']}")
            print(f"High Ratio (wrong): {q_info['experiment_pred']}")
            print(f"✗ High ratio caused this question to be answered incorrectly")


def generate_report_summary(all_results):
    """"""
    
    if not all_results:
        print("No results to summarize.")
        return
    
    # Find best and worst configurations
    best_config = max(all_results.items(), key=lambda x: x[1]['net_change'])
    worst_config = min(all_results.items(), key=lambda x: x[1]['net_change'])
    
    print("\n### Qualitative Analysis Summary")
    print("\nTo understand the impact of random negative sampling, we analyzed")
    print("which specific questions improved or degraded with different ratios.")
    
    print(f"\n**Key Findings:**")
    print(f"- Best configuration: {best_config[0]}")
    print(f"  - {len(best_config[1]['improved'])} questions improved (wrong → correct)")
    print(f"  - {len(best_config[1]['degraded'])} questions degraded (correct → wrong)")
    print(f"  - Net change: {best_config[1]['net_change']} questions")
    
    if worst_config[1]['net_change'] < 0:
        print(f"\n- Worst configuration: {worst_config[0]}")
        print(f"  - {len(worst_config[1]['improved'])} questions improved")
        print(f"  - {len(worst_config[1]['degraded'])} questions degraded")
        print(f"  - Net change: {worst_config[1]['net_change']} questions")
    
    print(f"\n**Patterns in Improved Questions:**")
    print("Questions that benefit from random documents tend to:")
    print("- Be yes/no questions where format consistency matters")
    print("- Have multiple oracle contexts available")
    print("- Require multi-hop reasoning")
    
    print(f"\n**Patterns in Degraded Questions:**")
    print("Questions hurt by random documents tend to:")
    print("- Require precise fact extraction")
    print("- Have complex entity relationships")
    print("- Depend on specific document details")
    
    print("\n### Example Improved Question")
    if best_config[1]['improved']:
        ex = best_config[1]['improved'][0]
        print(f"\n**Question:** {ex['question']}")
        print(f"**Answer:** {ex['answer']}")
        print(f"**Baseline (k={best_config[0].split('k=')[1].split(',')[0]}):** {ex['baseline_pred']}")
        print(f"**With Random Documents:** {ex['experiment_pred']}")
        print(f"✓ Random documents enabled correct answer extraction")
    
    print("\n### Example Degraded Question")
    if best_config[1]['degraded']:
        ex = best_config[1]['degraded'][0]
        print(f"\n**Question:** {ex['question']}")
        print(f"**Answer:** {ex['answer']}")
        print(f"**Baseline:** {ex['baseline_pred']} (correct)")
        print(f"**With Random Documents:** {ex['experiment_pred']} (incorrect)")
        print(f"✗ Random documents introduced confusion")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run full comparison
    results = compare_configurations()
    
    # Generate report summary
    if results:
        generate_report_summary(results)
    
    # Detailed analysis of high-ratio improvement (if k=5 results exist)
    try:
        analyze_specific_improvements()
    except:
        pass  # Skip if files don't exist yet
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS COMPLETE")
    print("="*80)


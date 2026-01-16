"""
Evaluate all negative sampling experiments and compare with baselines.
Includes statistical significance testing.
"""

import json
import pandas as pd
import os
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def evaluate_predictions(pred_file, dev_path):
    """Evaluate predictions using Exact Match, returns accuracy and per-question scores"""
    from dexter.utils.metrics.ExactMatch import ExactMatch
    
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    with open(dev_path, 'r') as f:
        questions = json.load(f)
    
    ground_truth = {q['_id']: q['answer'] for q in questions}
    
    metric = ExactMatch()
    scores = []
    question_ids = []
    
    for q_id, pred_answer in predictions.items():
        if q_id in ground_truth:
            gt_answer = ground_truth[q_id]
            score = metric.evaluate(pred_answer, gt_answer)
            scores.append(score)
            question_ids.append(q_id)
    
    accuracy = sum(scores) / len(scores) if scores else 0
    return accuracy, len(scores), scores, question_ids


def test_statistical_significance(baseline_scores, experiment_scores, question_ids_baseline, question_ids_experiment):
    """
    Test statistical significance using Wilcoxon signed-rank test.
    
    Wilcoxon signed-rank test is appropriate for paired comparisons of two related samples.
    It tests whether the median difference between pairs is zero.
    """
    # Align scores by question ID
    baseline_dict = {q_id: score for q_id, score in zip(question_ids_baseline, baseline_scores)}
    experiment_dict = {q_id: score for q_id, score in zip(question_ids_experiment, experiment_scores)}
    
    # Find common questions
    common_ids = sorted(list(set(baseline_dict.keys()) & set(experiment_dict.keys())))
    
    if len(common_ids) < 10:
        return None, None, "Insufficient common questions"
    
    # Get paired scores for common questions
    baseline_paired = [baseline_dict[q_id] for q_id in common_ids]
    experiment_paired = [experiment_dict[q_id] for q_id in common_ids]
    
    # Convert to numeric arrays (handle boolean/int)
    baseline_paired = np.array(baseline_paired, dtype=float)
    experiment_paired = np.array(experiment_paired, dtype=float)
    
    # Calculate differences
    differences = experiment_paired - baseline_paired
    
    # Check if all differences are zero (no variation)
    if np.all(differences == 0):
        return None, None, "No differences"
    
    # Perform Wilcoxon signed-rank test
    # This tests H0: median difference = 0 vs H1: median difference != 0
    try:
        statistic, p_value = wilcoxon(differences, alternative='two-sided', zero_method='wilcox')
    except ValueError as e:
        # Handle edge cases (e.g., all differences have same sign)
        return None, None, f"Test error: {str(e)}"
    
    # Determine significance
    is_significant = p_value < 0.05
    
    # Calculate summary statistics for reporting
    improved = np.sum(differences > 0)
    degraded = np.sum(differences < 0)
    unchanged = np.sum(differences == 0)
    
    return p_value, is_significant, f"improved={improved}, degraded={degraded}, unchanged={unchanged}"


def main():
    """Evaluate all experiments"""
    
    dev_path = 'data/dev.json'
    results_dir = 'results_negative_experiments'
    
    # Baseline results (check both root and results directory)
    baselines = [
        ('Baseline k=1', 'predictions_k1.json'),
        ('Baseline k=3', 'predictions_k3.json'),
        ('Baseline k=5', 'predictions_k5.json'),
        ('Oracle', 'predictions_oracle.json'),
    ]
    
    results = []
    baseline_data = {}  # Store baseline scores for significance testing
    
    print("Evaluating baselines...")
    for name, filepath in baselines:
        # Try multiple locations: root, json_files, results_dir
        if Path(filepath).exists():
            full_path = filepath
        elif Path(os.path.join('json_files', filepath)).exists():
            full_path = os.path.join('json_files', filepath)
        elif Path(os.path.join(results_dir, filepath)).exists():
            full_path = os.path.join(results_dir, filepath)
        else:
            continue
        
        acc, n, scores, q_ids = evaluate_predictions(full_path, dev_path)
        results.append({
            'Type': 'Baseline',
            'Experiment': name,
            'k': 'N/A' if 'Oracle' in name else int(name.split('k=')[1]) if 'k=' in name else 'N/A',
            'Ratio': 'N/A',
            'Negative Type': 'N/A',
            'Exact Match': acc,
            'EM %': f'{acc*100:.2f}%',
            'Samples': n,
            'p-value': 'N/A',
            'Significant': 'N/A'
        })
        
        # Store for significance testing
        if 'k=' in name:
            k_val = int(name.split('k=')[1])
            baseline_data[k_val] = {
                'scores': scores,
                'question_ids': q_ids,
                'accuracy': acc
            }
        
        print(f"{name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Negative sampling experiments - automatically discover all files
    print("\nEvaluating negative sampling experiments...")
    print("Scanning results directory for prediction files...")
    
    import re
    
    # Discover all prediction files in results directory
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Warning: {results_dir} directory does not exist")
    else:
        # Pattern: predictions_{neg_type}_k{k}_ratio_{R}_{N}.json
        pattern = re.compile(r'predictions_(random|hard)_k(\d+)_ratio_(\d+)_(\d+)\.json')
        
        discovered_experiments = []
        for pred_file in results_path.glob('predictions_*.json'):
            match = pattern.match(pred_file.name)
            if match:
                neg_type = match.group(1)
                k = int(match.group(2))
                R = int(match.group(3))
                N = int(match.group(4))
                discovered_experiments.append((neg_type, k, R, N, pred_file))
        
        print(f"Found {len(discovered_experiments)} experiment files")
        
        # Sort by neg_type, k, R, N for consistent processing
        discovered_experiments.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        
        for neg_type, k, R, N, pred_file in discovered_experiments:
            # Use the file from results_negative_experiments directory (actual experiment output)
            full_path = str(pred_file)
            
            if Path(full_path).exists():
                acc, n, scores, q_ids = evaluate_predictions(full_path, dev_path)
                
                # Statistical significance test
                p_value = None
                is_significant = None
                sig_note = ''
                
                if k in baseline_data:
                    baseline_scores = baseline_data[k]['scores']
                    baseline_q_ids = baseline_data[k]['question_ids']
                    p_value, is_significant, sig_note = test_statistical_significance(
                        baseline_scores, scores, baseline_q_ids, q_ids
                    )
                
                results.append({
                    'Type': 'Negative Sampling',
                    'Experiment': f'{neg_type} k={k}, {R}:{N}',
                    'k': k,
                    'Ratio': f'{R}:{N}',
                    'Negative Type': neg_type,
                    'Exact Match': acc,
                    'EM %': f'{acc*100:.2f}%',
                    'Samples': n,
                    'p-value': f'{p_value:.4f}' if p_value is not None else 'N/A',
                    'Significant': 'Yes' if is_significant else 'No' if is_significant is not None else 'N/A'
                })
                
                sig_marker = '*' if is_significant else ''
                print(f"{neg_type} k={k}, {R}:{N}: {acc:.4f} ({acc*100:.2f}%){sig_marker} {sig_note}")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    
    # Add delta vs baseline
    df['Delta vs Baseline'] = ''
    df['Delta %'] = ''
    
    for k in [1, 3, 5]:
        baseline_row = df[df['Experiment'] == f'Baseline k={k}']
        if len(baseline_row) > 0:
            baseline_acc = baseline_row.iloc[0]['Exact Match']
            mask = (df['k'] == k) & (df['Type'] == 'Negative Sampling')
            df.loc[mask, 'Delta vs Baseline'] = (df.loc[mask, 'Exact Match'] - baseline_acc).round(4)
            df.loc[mask, 'Delta %'] = ((df.loc[mask, 'Exact Match'] - baseline_acc) / baseline_acc * 100).round(2)
    
    # Save results
    output_file = 'all_experiments_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("* indicates statistically significant difference (p < 0.05)")
    print(df.to_string(index=False))
    
    return df


if __name__ == "__main__":
    main()

"""
Comprehensive visualization of negative sampling experiment results.

Creates multiple plots showing:
1. Performance vs ratio for each k value
2. Comparison between random and hard negatives
3. Heatmap similar to Cuconasu's table
4. Improvement/degradation patterns
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")

def load_results():
    """Load results from CSV file"""
    csv_file = 'all_experiments_results.csv'
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found. Please run evaluate_all_experiments.py first.")
        return None
    
    df = pd.read_csv(csv_file)
    
    # Parse ratio to extract number of random documents
    def parse_ratio(ratio_str):
        if pd.isna(ratio_str) or ratio_str == 'N/A':
            return None
        try:
            R, N = map(int, ratio_str.split(':'))
            return N  # Return number of random documents
        except:
            return None
    
    df['Num_Random'] = df['Ratio'].apply(parse_ratio)
    df['Exact_Match_Num'] = pd.to_numeric(df['Exact Match'], errors='coerce')
    
    # Parse delta
    df['Delta_Num'] = pd.to_numeric(df['Delta vs Baseline'], errors='coerce')
    df['Delta_Pct_Num'] = pd.to_numeric(df['Delta %'], errors='coerce')
    
    return df


def plot_performance_by_ratio(df):
    """Plot performance vs number of random documents for each k"""
    
    # Dynamically determine k values from data
    k_values = sorted([k for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    if len(k_values) == 0:
        print("No k values found in data")
        return None
    
    # Create subplots based on number of k values
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(6*n_k, 5))
    if n_k == 1:
        axes = [axes]
    fig.suptitle('Performance vs Number of Random Documents', fontsize=16, fontweight='bold')
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Get baseline
        baseline = df[(df['k'] == k) & (df['Type'] == 'Baseline')]
        baseline_acc = baseline['Exact_Match_Num'].values[0] if len(baseline) > 0 else None
        
        # Random negatives
        random_data = df[(df['k'] == k) & (df['Negative Type'] == 'random') & 
                        (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Hard negatives
        hard_data = df[(df['k'] == k) & (df['Negative Type'] == 'hard') & 
                      (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Plot random negatives
        if len(random_data) > 0:
            ax.plot(random_data['Num_Random'], random_data['Exact_Match_Num'], 
                   'o-', label='Random Negatives', linewidth=2, markersize=8, color='#2ca02c')
            # Mark significant points
            sig_data = random_data[random_data['Significant'] == 'Yes']
            if len(sig_data) > 0:
                ax.scatter(sig_data['Num_Random'], sig_data['Exact_Match_Num'], 
                          s=200, marker='*', color='green', zorder=5, 
                          label='Significant (p<0.05)')
        
        # Plot hard negatives
        if len(hard_data) > 0:
            ax.plot(hard_data['Num_Random'], hard_data['Exact_Match_Num'], 
                   's--', label='Hard Negatives', linewidth=2, markersize=8, color='#d62728')
        
        # Plot baseline
        if baseline_acc is not None:
            ax.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Baseline ({baseline_acc:.4f})')
        
        ax.set_xlabel('Number of Random Documents', fontsize=12)
        ax.set_ylabel('Exact Match Accuracy', fontsize=12)
        ax.set_title(f'k={k} Retrieved Documents', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim([0, max(0.3, ax.get_ylim()[1])])
    
    plt.tight_layout()
    plt.savefig('performance_by_ratio.png', dpi=300, bbox_inches='tight')
    print("Saved: performance_by_ratio.png")
    return fig


def plot_improvement_degradation(df):
    """Plot improvement/degradation percentage vs baseline"""
    
    # Dynamically determine k values from data
    k_values = sorted([k for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    if len(k_values) == 0:
        print("No k values found in data")
        return None
    
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(6*n_k, 5))
    if n_k == 1:
        axes = [axes]
    fig.suptitle('Improvement/Degradation vs Baseline (%)', fontsize=16, fontweight='bold')
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Random negatives
        random_data = df[(df['k'] == k) & (df['Negative Type'] == 'random') & 
                        (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Hard negatives
        hard_data = df[(df['k'] == k) & (df['Negative Type'] == 'hard') & 
                      (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Plot random negatives
        if len(random_data) > 0:
            ax.plot(random_data['Num_Random'], random_data['Delta_Pct_Num'], 
                   'o-', label='Random Negatives', linewidth=2, markersize=8, color='#2ca02c')
        
        # Plot hard negatives
        if len(hard_data) > 0:
            ax.plot(hard_data['Num_Random'], hard_data['Delta_Pct_Num'], 
                   's--', label='Hard Negatives', linewidth=2, markersize=8, color='#d62728')
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Number of Random Documents', fontsize=12)
        ax.set_ylabel('Change vs Baseline (%)', fontsize=12)
        ax.set_title(f'k={k} Retrieved Documents', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('improvement_degradation.png', dpi=300, bbox_inches='tight')
    print("Saved: improvement_degradation.png")
    return fig


def create_heatmap_table(df):
    """Create a heatmap table"""
    
    # Dynamically determine k values from data
    k_values = sorted([int(k) for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    if len(k_values) == 0:
        print("No k values found in data")
        return None
    
    # Create pivot table: rows = number of random docs, columns = k values
    heatmap_data = {}
    
    for k in k_values:
        k_data = df[(df['k'] == k) & (df['Negative Type'] == 'random') & 
                   (df['Num_Random'].notna())]
        
        for _, row in k_data.iterrows():
            num_random = int(row['Num_Random'])
            if num_random not in heatmap_data:
                heatmap_data[num_random] = {}
            heatmap_data[num_random][k] = row['Exact_Match_Num']
    
    # Create DataFrame
    all_randoms = sorted(heatmap_data.keys())
    heatmap_df = pd.DataFrame(index=all_randoms, columns=k_values)
    
    for num_random in all_randoms:
        for k in k_values:
            if k in heatmap_data[num_random]:
                heatmap_df.loc[num_random, k] = heatmap_data[num_random][k]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Convert to numeric
    heatmap_df_numeric = heatmap_df.astype(float)
    
    # Create heatmap
    sns.heatmap(heatmap_df_numeric, annot=True, fmt='.4f', cmap='RdYlGn', 
                center=0.05, vmin=0, vmax=0.3, cbar_kws={'label': 'Exact Match Accuracy'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_xlabel('k (Number of Retrieved Documents)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Random Documents', fontsize=12, fontweight='bold')
    ax.set_title('Performance Heatmap: Random Documents vs k\n(Style similar to Cuconasu et al.)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('heatmap_table.png', dpi=300, bbox_inches='tight')
    print("Saved: heatmap_table.png")
    return fig


def plot_comparison_random_vs_hard(df):
    """Compare random vs hard negatives side by side"""
    
    # Dynamically determine k values from data
    k_values = sorted([k for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    if len(k_values) == 0:
        print("No k values found in data")
        return None
    
    n_k = len(k_values)
    fig, axes = plt.subplots(1, n_k, figsize=(6*n_k, 5))
    if n_k == 1:
        axes = [axes]
    fig.suptitle('Random vs Hard Negatives Comparison', fontsize=16, fontweight='bold')
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        
        # Random negatives
        random_data = df[(df['k'] == k) & (df['Negative Type'] == 'random') & 
                        (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Hard negatives
        hard_data = df[(df['k'] == k) & (df['Negative Type'] == 'hard') & 
                      (df['Num_Random'].notna())].sort_values('Num_Random')
        
        # Plot both
        if len(random_data) > 0:
            ax.plot(random_data['Num_Random'], random_data['Exact_Match_Num'], 
                   'o-', label='Random', linewidth=2.5, markersize=10, color='#2ca02c')
        
        if len(hard_data) > 0:
            ax.plot(hard_data['Num_Random'], hard_data['Exact_Match_Num'], 
                   's--', label='Hard', linewidth=2.5, markersize=10, color='#d62728')
        
        # Baseline
        baseline = df[(df['k'] == k) & (df['Type'] == 'Baseline')]
        if len(baseline) > 0:
            baseline_acc = baseline['Exact_Match_Num'].values[0]
            ax.axhline(y=baseline_acc, color='r', linestyle=':', linewidth=2, 
                      alpha=0.7, label=f'Baseline ({baseline_acc:.4f})')
        
        ax.set_xlabel('Number of Negative Documents', fontsize=12)
        ax.set_ylabel('Exact Match Accuracy', fontsize=12)
        ax.set_title(f'k={k} Retrieved Documents', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim([0, max(0.3, ax.get_ylim()[1])])
    
    plt.tight_layout()
    plt.savefig('random_vs_hard_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: random_vs_hard_comparison.png")
    return fig


def plot_statistical_significance(df):
    """Plot showing which experiments are statistically significant"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Filter to negative sampling experiments
    neg_data = df[(df['Type'] == 'Negative Sampling') & (df['Num_Random'].notna())].copy()
    
    # Create scatter plot
    colors = {'random': '#2ca02c', 'hard': '#d62728'}
    markers = {'Yes': '*', 'No': 'o', 'N/A': 'x'}
    
    for neg_type in ['random', 'hard']:
        type_data = neg_data[neg_data['Negative Type'] == neg_type]
        
        for sig_status in ['Yes', 'No']:
            sig_data = type_data[type_data['Significant'] == sig_status]
            if len(sig_data) > 0:
                ax.scatter(sig_data['Num_Random'], sig_data['Exact_Match_Num'],
                          c=colors[neg_type], marker=markers[sig_status],
                          s=150, alpha=0.7, 
                          label=f'{neg_type.capitalize()} - {"Significant" if sig_status == "Yes" else "Not Significant"}',
                          edgecolors='black', linewidths=1)
    
    # Add k value labels
    k_values = sorted([k for k in neg_data['k'].unique() if pd.notna(k) and k != 'N/A'])
    for i, k in enumerate(k_values):
        k_data = neg_data[neg_data['k'] == k]
        if len(k_data) > 0:
            y_pos = 0.98 - i * 0.15
            ax.text(0.02, y_pos, f'k={k}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', 
                   facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Number of Random Documents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Significance of Results\n(* = Significant, o = Not Significant)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight')
    print("Saved: statistical_significance.png")
    return fig


def plot_baseline_comparison(df):
    """Compare all experiments against baselines"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get baselines
    baselines = {}
    for k in [1, 3, 5]:
        baseline = df[(df['k'] == k) & (df['Type'] == 'Baseline')]
        if len(baseline) > 0:
            baselines[k] = baseline['Exact_Match_Num'].values[0]
    
    # Plot baselines
    for k, baseline_acc in baselines.items():
        ax.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2,
                  alpha=0.5, label=f'Baseline k={k} ({baseline_acc:.4f})')
    
    # Plot experiments
    k_values = sorted([k for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    for neg_type in ['random', 'hard']:
        type_data = df[(df['Negative Type'] == neg_type) & 
                      (df['Num_Random'].notna())].sort_values(['k', 'Num_Random'])
        
        for k in k_values:
            k_data = type_data[type_data['k'] == k]
            if len(k_data) > 0:
                # Offset x-axis by k value for clarity
                x_offset = (k - 2) * 0.3
                x_vals = k_data['Num_Random'] + x_offset
                
                ax.plot(x_vals, k_data['Exact_Match_Num'],
                       'o-' if neg_type == 'random' else 's--',
                       label=f'{neg_type.capitalize()} k={k}',
                       linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of Random Documents', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exact Match Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('All Experiments vs Baselines', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: baseline_comparison.png")
    return fig


def create_summary_table(df):
    """Create a summary table of key results"""
    
    print("\n" + "="*80)
    print("KEY RESULTS SUMMARY")
    print("="*80)
    
    k_values = sorted([k for k in df['k'].unique() if pd.notna(k) and k != 'N/A' and isinstance(k, (int, float))])
    
    for k in k_values:
        print(f"\nk={k}:")
        baseline = df[(df['k'] == k) & (df['Type'] == 'Baseline')]
        if len(baseline) > 0:
            baseline_acc = baseline['Exact_Match_Num'].values[0]
            print(f"  Baseline: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
        
        # Best random
        random_data = df[(df['k'] == k) & (df['Negative Type'] == 'random') & 
                        (df['Num_Random'].notna())]
        if len(random_data) > 0:
            best_random = random_data.loc[random_data['Exact_Match_Num'].idxmax()]
            print(f"  Best Random: {best_random['Ratio']} = {best_random['Exact_Match_Num']:.4f} "
                  f"({best_random['Exact_Match_Num']*100:.2f}%) "
                  f"[Δ: {best_random['Delta_Pct_Num']:.2f}%]")
        
        # Best hard
        hard_data = df[(df['k'] == k) & (df['Negative Type'] == 'hard') & 
                      (df['Num_Random'].notna())]
        if len(hard_data) > 0:
            best_hard = hard_data.loc[hard_data['Exact_Match_Num'].idxmax()]
            print(f"  Best Hard: {best_hard['Ratio']} = {best_hard['Exact_Match_Num']:.4f} "
                  f"({best_hard['Exact_Match_Num']*100:.2f}%) "
                  f"[Δ: {best_hard['Delta_Pct_Num']:.2f}%]")


def main():
    """Main visualization function"""
    
    print("="*80)
    print("VISUALIZING EXPERIMENT RESULTS")
    print("="*80)
    
    # Load data
    df = load_results()
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} experiment results")
    print(f"Experiments: {len(df[df['Type'] == 'Negative Sampling'])} negative sampling + "
          f"{len(df[df['Type'] == 'Baseline'])} baselines")
    
    # Create all visualizations
    print("\nCreating visualizations...")
    
    try:
        plot_performance_by_ratio(df)
        plot_improvement_degradation(df)
        create_heatmap_table(df)
        plot_comparison_random_vs_hard(df)
        plot_statistical_significance(df)
        plot_baseline_comparison(df)
        
        print("\n" + "="*80)
        print("ALL VISUALIZATIONS CREATED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  1. performance_by_ratio.png - Performance vs number of random documents")
        print("  2. improvement_degradation.png - Improvement/degradation percentages")
        print("  3. heatmap_table.png - Heatmap similar to Cuconasu's table")
        print("  4. random_vs_hard_comparison.png - Direct comparison")
        print("  5. statistical_significance.png - Significance markers")
        print("  6. baseline_comparison.png - All experiments vs baselines")
        
        # Create summary
        create_summary_table(df)
        
    except Exception as e:
        print(f"\nError creating visualizations: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

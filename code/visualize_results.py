"""
visualize_results.py

Visualize experiment results from run_algorithms.py output.
Creates comprehensive visualizations comparing algorithms across benchmarks.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def load_results(json_path: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def categorize_results(results: List[Dict]) -> Dict:
    """Categorize results by algorithm and benchmark."""
    categorized = {}
    for r in results:
        algo = r['algorithm']
        bench = r['benchmark']
        key = (algo, bench)
        if key not in categorized:
            categorized[key] = []
        categorized[key].append(r)
    return categorized


def create_success_rate_heatmap(results: List[Dict], output_path: str):
    """Create a heatmap showing success rates by algorithm and benchmark."""
    algorithms = sorted(set(r['algorithm'] for r in results))
    benchmarks = sorted(set(r['benchmark'] for r in results))
    
    success_matrix = np.zeros((len(algorithms), len(benchmarks)))
    
    for i, algo in enumerate(algorithms):
        for j, bench in enumerate(benchmarks):
            matching = [r for r in results if r['algorithm'] == algo and r['benchmark'] == bench]
            if matching:
                successes = sum(1 for r in matching if r['status'] == 'success')
                success_matrix[i, j] = successes / len(matching)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(success_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.set_yticklabels(algorithms)
    
    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(benchmarks)):
            text = ax.text(j, i, f'{success_matrix[i, j]:.0%}',
                          ha="center", va="center", color="black" if success_matrix[i, j] > 0.5 else "white",
                          fontweight='bold')
    
    ax.set_title('Success Rate by Algorithm and Benchmark', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Algorithm', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_training_time_comparison(results: List[Dict], output_path: str):
    """Create a bar chart comparing training times."""
    algorithms = sorted(set(r['algorithm'] for r in results))
    benchmarks = sorted(set(r['benchmark'] for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        algo_times = []
        algo_names = []
        
        for algo in algorithms:
            matching = [r for r in results if r['algorithm'] == algo and r['benchmark'] == bench and r['status'] == 'success']
            if matching:
                avg_time = np.mean([r['training_time'] for r in matching])
                algo_times.append(avg_time)
                algo_names.append(algo)
        
        if algo_times:
            bars = ax.bar(algo_names, algo_times, color=sns.color_palette("husl", len(algo_names)))
            ax.set_title(f'{bench}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Training Time (seconds)', fontsize=10)
            ax.set_xlabel('Algorithm', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}s',
                       ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Training Time Comparison by Algorithm and Benchmark', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_algorithm_comparison(results: List[Dict], output_path: str):
    """Create a comprehensive comparison across all algorithms."""
    algorithms = sorted(set(r['algorithm'] for r in results))
    benchmarks = sorted(set(r['benchmark'] for r in results))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(benchmarks))
    width = 0.13
    multiplier = 0
    
    colors = sns.color_palette("Set2", len(algorithms))
    
    for algo in algorithms:
        success_counts = []
        for bench in benchmarks:
            matching = [r for r in results if r['algorithm'] == algo and r['benchmark'] == bench]
            successes = sum(1 for r in matching if r['status'] == 'success')
            success_counts.append(successes)
        
        offset = width * multiplier
        bars = ax.bar(x + offset, success_counts, width, label=algo, color=colors[multiplier])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Number of Successful Experiments', fontsize=12)
    ax.set_title('Algorithm Performance Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper left', ncol=len(algorithms))
    ax.set_ylim(0, max([sum(1 for r in results if r['algorithm'] == algo and r['benchmark'] == bench and r['status'] == 'success')
                        for algo in algorithms for bench in benchmarks]) + 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_status_summary(results: List[Dict], output_path: str):
    """Create a summary pie chart and bar chart of experiment statuses."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart
    status_counts = {}
    for r in results:
        status = r['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    colors_pie = {'success': '#2ecc71', 'failed': '#e74c3c', 'timeout': '#f39c12', 'error': '#9b59b6'}
    pie_colors = [colors_pie.get(status, '#95a5a6') for status in status_counts.keys()]
    
    ax1.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.1f%%',
           colors=pie_colors, startangle=90)
    ax1.set_title('Overall Experiment Status Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart by algorithm
    algorithms = sorted(set(r['algorithm'] for r in results))
    status_by_algo = {algo: {'success': 0, 'failed': 0, 'timeout': 0, 'error': 0} 
                     for algo in algorithms}
    
    for r in results:
        algo = r['algorithm']
        status = r['status']
        if status in status_by_algo[algo]:
            status_by_algo[algo][status] += 1
    
    x = np.arange(len(algorithms))
    width = 0.2
    multiplier = 0
    
    for status in ['success', 'failed', 'timeout', 'error']:
        counts = [status_by_algo[algo].get(status, 0) for algo in algorithms]
        offset = width * multiplier
        ax2.bar(x + offset, counts, width, label=status, color=colors_pie.get(status, '#95a5a6'))
        multiplier += 1
    
    ax2.set_xlabel('Algorithm', fontsize=12)
    ax2.set_ylabel('Number of Experiments', fontsize=12)
    ax2.set_title('Experiment Status by Algorithm', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(algorithms)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_detailed_table(results: List[Dict], output_path: str):
    """Create a detailed results table visualization."""
    algorithms = sorted(set(r['algorithm'] for r in results))
    benchmarks = sorted(set(r['benchmark'] for r in results))
    
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Algorithm'] + benchmarks + ['Total Success']
    
    for algo in algorithms:
        row = [algo]
        total_success = 0
        for bench in benchmarks:
            matching = [r for r in results if r['algorithm'] == algo and r['benchmark'] == bench]
            if matching:
                successes = sum(1 for r in matching if r['status'] == 'success')
                total = len(matching)
                total_success += successes
                cell_text = f"{successes}/{total}"
                if successes == total:
                    cell_text += " ✓"
                row.append(cell_text)
            else:
                row.append("N/A")
        row.append(f"{total_success}/{len(benchmarks)}")
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=headers,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code cells
    for i in range(len(table_data)):
        for j in range(1, len(headers)):
            cell = table[(i+1, j)]
            if j < len(headers) - 1:  # Not the total column
                text = cell.get_text().get_text()
                if '✓' in text:
                    cell.set_facecolor('#d4edda')
                elif '/' in text and '0/' in text:
                    cell.set_facecolor('#f8d7da')
                else:
                    cell.set_facecolor('#fff3cd')
            else:
                # Total column
                text = cell.get_text().get_text()
                if text.endswith(f'/{len(benchmarks)}'):
                    cell.set_facecolor('#d1ecf1')
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#343a40')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('Detailed Results Table', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results-file', type=str, 
                       default='results/results_20260203_121853.json',
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        # Try to find latest results file
        results_dir = Path('results')
        json_files = list(results_dir.glob('results_*.json'))
        if json_files:
            results_path = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"Using latest results file: {results_path}")
        else:
            raise FileNotFoundError(f"Results file not found: {args.results_file}")
    
    results = load_results(str(results_path))
    print(f"Loaded {len(results)} experiment results from {results_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_success_rate_heatmap(results, str(output_dir / 'success_rate_heatmap.png'))
    create_training_time_comparison(results, str(output_dir / 'training_time_comparison.png'))
    create_algorithm_comparison(results, str(output_dir / 'algorithm_comparison.png'))
    create_status_summary(results, str(output_dir / 'status_summary.png'))
    create_detailed_table(results, str(output_dir / 'detailed_results_table.png'))
    
    print(f"\nAll visualizations saved to {output_dir}/")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"Total experiments: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    
    if successful > 0:
        avg_time = np.mean([r['training_time'] for r in results if r['status'] == 'success'])
        print(f"Average training time (successful): {avg_time:.2f}s")
    
    print("="*60)


if __name__ == "__main__":
    main()

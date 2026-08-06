"""
visualize_training_curves.py

Visualize reward and loss curves from TensorBoard logs.
Extracts episodic_return and td_loss metrics and creates comprehensive visualizations.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10


def parse_run_name(run_name: str) -> Tuple[str, str]:
    """Parse run name to extract benchmark and algorithm."""
    # Format: Benchmark__composable_feedback_cleanrl__optAlgorithm__seedSeed__timestamp
    match = re.match(r'([^_]+)-v\d+__composable_feedback_cleanrl__opt(\w+)__seed\d+__\d+', run_name)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_tensorboard_data(runs_dir: Path) -> Dict:
    """Load all TensorBoard data from runs directory."""
    data = {}
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        benchmark, algorithm = parse_run_name(run_dir.name)
        if benchmark is None or algorithm is None:
            continue
        
        # Load event file
        event_files = list(run_dir.glob("events.out.tfevents.*"))
        if not event_files:
            continue
        
        try:
            ea = EventAccumulator(str(run_dir))
            ea.Reload()
            
            # Extract scalar data
            scalar_tags = ea.Tags()['scalars']
            
            if 'charts/episodic_return' in scalar_tags:
                episodic_return = ea.Scalars('charts/episodic_return')
                returns = [(s.step, s.value) for s in episodic_return]
            else:
                returns = []
            
            if 'losses/td_loss' in scalar_tags:
                td_loss = ea.Scalars('losses/td_loss')
                losses = [(s.step, s.value) for s in td_loss]
            else:
                losses = []
            
            if returns or losses:
                key = (benchmark, algorithm)
                if key not in data:
                    data[key] = {'returns': [], 'losses': []}
                
                if returns:
                    data[key]['returns'].extend(returns)
                if losses:
                    data[key]['losses'].extend(losses)
        
        except Exception as e:
            print(f"Warning: Could not load {run_dir.name}: {e}")
            continue
    
    return data


def smooth_curve(values: List[float], window: int = 100) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    if len(values) < window:
        return np.array(values)
    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
    # Pad beginning to maintain length
    padding = window - 1
    return np.concatenate([values[:padding], smoothed])


def create_reward_curves(data: Dict, output_path: str):
    """Create reward/episodic return curves for all algorithms and benchmarks."""
    benchmarks = sorted(set(k[0] for k in data.keys()))
    algorithms = sorted(set(k[1] for k in data.keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(algorithms))
    algo_colors = {algo: colors[i] for i, algo in enumerate(algorithms)}
    
    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        
        for algo in algorithms:
            key = (bench, algo)
            if key not in data or not data[key]['returns']:
                continue
            
            returns = data[key]['returns']
            if not returns:
                continue
            
            # Sort by step
            returns.sort(key=lambda x: x[0])
            steps = [r[0] for r in returns]
            values = [r[1] for r in returns]
            
            # Smooth the curve
            smoothed = smooth_curve(values, window=min(50, len(values)//10))
            
            ax.plot(steps[:len(smoothed)], smoothed, label=algo, color=algo_colors[algo], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Episodic Return', fontsize=11)
        ax.set_title(f'{bench}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Episodic Return (Reward) Curves by Algorithm and Benchmark', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_loss_curves(data: Dict, output_path: str):
    """Create TD loss curves for all algorithms and benchmarks."""
    benchmarks = sorted(set(k[0] for k in data.keys()))
    algorithms = sorted(set(k[1] for k in data.keys()))
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(algorithms))
    algo_colors = {algo: colors[i] for i, algo in enumerate(algorithms)}
    
    for idx, bench in enumerate(benchmarks):
        ax = axes[idx]
        
        for algo in algorithms:
            key = (bench, algo)
            if key not in data or not data[key]['losses']:
                continue
            
            losses = data[key]['losses']
            if not losses:
                continue
            
            # Sort by step
            losses.sort(key=lambda x: x[0])
            steps = [l[0] for l in losses]
            values = [l[1] for l in losses]
            
            # Smooth the curve
            smoothed = smooth_curve(values, window=min(50, len(values)//10))
            
            ax.plot(steps[:len(smoothed)], smoothed, label=algo, color=algo_colors[algo], alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('TD Loss', fontsize=11)
        ax.set_title(f'{bench}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # Log scale for loss
    
    plt.suptitle('TD Loss Curves by Algorithm and Benchmark', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_algorithm_comparison_rewards(data: Dict, output_path: str):
    """Create side-by-side comparison of reward curves for each algorithm."""
    benchmarks = sorted(set(k[0] for k in data.keys()))
    algorithms = sorted(set(k[1] for k in data.keys()))
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = sns.color_palette("Set2", len(benchmarks))
    bench_colors = {bench: colors[i] for i, bench in enumerate(benchmarks)}
    
    for idx, algo in enumerate(algorithms):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        for bench in benchmarks:
            key = (bench, algo)
            if key not in data or not data[key]['returns']:
                continue
            
            returns = data[key]['returns']
            if not returns:
                continue
            
            returns.sort(key=lambda x: x[0])
            steps = [r[0] for r in returns]
            values = [r[1] for r in returns]
            
            smoothed = smooth_curve(values, window=min(50, len(values)//10))
            ax.plot(steps[:len(smoothed)], smoothed, label=bench, color=bench_colors[bench], 
                   alpha=0.8, linewidth=2)
        
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel('Episodic Return', fontsize=10)
        ax.set_title(f'Algorithm: {algo}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(algorithms), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Reward Curves: Algorithm Comparison Across Benchmarks', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_final_performance_comparison(data: Dict, output_path: str):
    """Create a comparison of final performance (last N episodes) for each algorithm."""
    benchmarks = sorted(set(k[0] for k in data.keys()))
    algorithms = sorted(set(k[1] for k in data.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(benchmarks))
    width = 0.13
    multiplier = 0
    
    colors = sns.color_palette("Set2", len(algorithms))
    
    for algo in algorithms:
        final_returns = []
        for bench in benchmarks:
            key = (bench, algo)
            if key not in data or not data[key]['returns']:
                final_returns.append(0)
                continue
            
            returns = data[key]['returns']
            if not returns:
                final_returns.append(0)
                continue
            
            returns.sort(key=lambda x: x[0])
            # Get last 10% of episodes
            last_n = max(1, len(returns) // 10)
            final_values = [r[1] for r in returns[-last_n:]]
            avg_final = np.mean(final_values) if final_values else 0
            final_returns.append(avg_final)
        
        offset = width * multiplier
        bars = ax.bar(x + offset, final_returns, width, label=algo, color=colors[multiplier])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('Average Final Episodic Return', fontsize=12)
    ax.set_title('Final Performance Comparison (Last 10% of Training)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper left', ncol=len(algorithms))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training curves from TensorBoard logs')
    parser.add_argument('--runs-dir', type=str, default='code/runs',
                       help='Directory containing TensorBoard run logs')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save visualizations')
    parser.add_argument('--algorithms', type=str, default=None,
                       help='Comma-separated list of algorithms to include (e.g., a,d). If not specified, all algorithms are included.')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    
    print(f"Loading TensorBoard data from {runs_dir}...")
    data = load_tensorboard_data(runs_dir)
    
    # Filter by algorithms if specified
    if args.algorithms:
        algo_filter = [a.strip() for a in args.algorithms.split(",") if a.strip()]
        data = {k: v for k, v in data.items() if k[1] in algo_filter}
        print(f"Filtered to algorithms: {algo_filter}")
    
    if not data:
        print("No TensorBoard data found!")
        return
    
    print(f"Loaded data for {len(data)} algorithm-benchmark combinations")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    create_reward_curves(data, str(output_dir / 'reward_curves.png'))
    create_loss_curves(data, str(output_dir / 'loss_curves.png'))
    create_algorithm_comparison_rewards(data, str(output_dir / 'algorithm_reward_comparison.png'))
    create_final_performance_comparison(data, str(output_dir / 'final_performance_comparison.png'))
    
    print(f"\nAll visualizations saved to {output_dir}/")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    for (bench, algo), metrics in sorted(data.items()):
        num_returns = len(metrics['returns'])
        num_losses = len(metrics['losses'])
        print(f"{bench} - {algo}: {num_returns} return points, {num_losses} loss points")
    print("="*60)


if __name__ == "__main__":
    main()

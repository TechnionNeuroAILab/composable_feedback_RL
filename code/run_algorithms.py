"""
run_algorithms.py

Script to systematically test all 6 algorithms on 4 simple benchmarks:
- Algorithms: standard, a, b, c, c_local, d
- Benchmarks: CartPole-v1, MountainCar-v0, Acrobot-v1, FrozenLake-v1

Usage:
    python run_algorithms.py [--algorithms standard,a,b] [--benchmarks CartPole-v1] [--seeds 1,2,3] [--total-timesteps 200000]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Configuration
ALGORITHMS = ["standard", "a", "b", "c", "c_local", "d"]
BENCHMARKS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "FrozenLake-v1"]
DEFAULT_TIMESTEPS = 200_000
DEFAULT_SEEDS = [1]


def run_experiment(
    algorithm: str,
    env_id: str,
    seed: int,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    script_path: str = "composable_feedback_cleanrl.py",
    timeout: int = 3600,
) -> Dict:
    """
    Run a single experiment by calling composable_feedback_cleanrl.py via subprocess.
    
    Args:
        algorithm: Algorithm option (standard, a, b, c, c_local, d)
        env_id: Environment ID (e.g., CartPole-v1)
        seed: Random seed
        total_timesteps: Total training timesteps
        script_path: Path to the composable_feedback_cleanrl.py script
        timeout: Maximum time in seconds before killing the process
        
    Returns:
        Dictionary with experiment results and metadata
    """
    start_time = time.time()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    script_full_path = script_dir / script_path
    
    # Build command
    cmd = [
        sys.executable,
        str(script_full_path),
        "--option", algorithm,
        "--env-id", env_id,
        "--seed", str(seed),
        "--total-timesteps", str(total_timesteps),
    ]
    
    result = {
        "algorithm": algorithm,
        "benchmark": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "status": "unknown",
        "training_time": 0.0,
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "error": None,
    }
    
    try:
        process_result = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        result["return_code"] = process_result.returncode
        result["stdout"] = process_result.stdout
        result["stderr"] = process_result.stderr
        
        if process_result.returncode == 0:
            result["status"] = "success"
            # Try to extract final episodic return from stdout
            # Look for patterns like "episodic_return=200.0"
            try:
                lines = process_result.stdout.split("\n")
                for line in reversed(lines):
                    if "episodic_return" in line.lower():
                        # Try to extract number
                        import re
                        match = re.search(r"episodic_return[=:]\s*([\d.]+)", line, re.IGNORECASE)
                        if match:
                            result["final_return"] = float(match.group(1))
                            break
            except Exception:
                pass
        else:
            result["status"] = "failed"
            result["error"] = f"Process returned non-zero exit code: {process_result.returncode}"
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Experiment exceeded timeout of {timeout} seconds"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    result["training_time"] = time.time() - start_time
    return result


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run all algorithms on benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(ALGORITHMS),
        help="Comma-separated list of algorithms to test",
    )
    
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=",".join(BENCHMARKS),
        help="Comma-separated list of benchmarks to test",
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="1",
        help="Comma-separated list of seeds or single number",
    )
    
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps per experiment",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per experiment in seconds",
    )
    
    parser.add_argument(
        "--script-path",
        type=str,
        default="composable_feedback_cleanrl.py",
        help="Path to composable_feedback_cleanrl.py script",
    )
    
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> List[int]:
    """Parse seeds string into list of integers."""
    seeds = []
    for s in seeds_str.split(","):
        s = s.strip()
        try:
            seeds.append(int(s))
        except ValueError:
            print(f"Warning: Could not parse seed '{s}', skipping")
    return seeds if seeds else DEFAULT_SEEDS


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_path / f"results_{timestamp}.json"
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {json_path}")
    return json_path


def print_summary_table(results: List[Dict]):
    """Print a summary table of results."""
    if not results:
        print("No results to display")
        return
    
    # Group by algorithm and benchmark
    summary = {}
    for r in results:
        key = (r["algorithm"], r["benchmark"])
        if key not in summary:
            summary[key] = {
                "returns": [],
                "times": [],
                "successes": 0,
                "failures": 0,
            }
        
        if r["status"] == "success":
            summary[key]["successes"] += 1
            if "final_return" in r:
                summary[key]["returns"].append(r["final_return"])
        else:
            summary[key]["failures"] += 1
        
        summary[key]["times"].append(r["training_time"])
    
    # Print table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    
    # Header
    benchmarks = sorted(set(r["benchmark"] for r in results))
    algorithms = sorted(set(r["algorithm"] for r in results))
    
    header = f"{'Algorithm':<15}"
    for bench in benchmarks:
        header += f" {bench:<20}"
    header += " Status"
    print(header)
    print("-" * 100)
    
    # Rows
    for algo in algorithms:
        row = f"{algo:<15}"
        for bench in benchmarks:
            key = (algo, bench)
            if key in summary:
                s = summary[key]
                if s["returns"]:
                    avg_return = sum(s["returns"]) / len(s["returns"])
                    avg_time = sum(s["times"]) / len(s["times"])
                    cell = f"R:{avg_return:.1f} T:{avg_time:.0f}s"
                else:
                    cell = "No data"
                if s["failures"] > 0:
                    cell += f" ({s['failures']}F)"
            else:
                cell = "Not run"
            row += f" {cell:<20}"
        
        # Overall status
        algo_results = [r for r in results if r["algorithm"] == algo]
        successes = sum(1 for r in algo_results if r["status"] == "success")
        total = len(algo_results)
        row += f" {successes}/{total}"
        print(row)
    
    print("=" * 100)
    print("\nLegend: R=Average Return, T=Average Time, F=Failures")


def main():
    """Main function to run all experiments."""
    args = parse_args()
    
    # Parse arguments
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    seeds = parse_seeds(args.seeds)
    
    # Validate
    invalid_algorithms = [a for a in algorithms if a not in ALGORITHMS]
    if invalid_algorithms:
        print(f"Warning: Invalid algorithms: {invalid_algorithms}")
        algorithms = [a for a in algorithms if a in ALGORITHMS]
    
    if not algorithms:
        print("Error: No valid algorithms specified")
        return
    
    if not benchmarks:
        print("Error: No benchmarks specified")
        return
    
    # Print configuration
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"Algorithms: {algorithms}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Seeds: {seeds}")
    print(f"Total timesteps per experiment: {args.total_timesteps:,}")
    print(f"Total experiments: {len(algorithms) * len(benchmarks) * len(seeds)}")
    print("=" * 80)
    print()
    
    # Run experiments
    all_results = []
    total_experiments = len(algorithms) * len(benchmarks) * len(seeds)
    
    # Create progress bar if tqdm is available
    if tqdm:
        pbar = tqdm(total=total_experiments, desc="Running experiments")
    else:
        pbar = None
    
    experiment_num = 0
    for algorithm in algorithms:
        for benchmark in benchmarks:
            for seed in seeds:
                experiment_num += 1
                
                if pbar:
                    pbar.set_description(f"{algorithm} on {benchmark} (seed {seed})")
                
                print(f"\n[{experiment_num}/{total_experiments}] Running {algorithm} on {benchmark} (seed {seed})...")
                
                result = run_experiment(
                    algorithm=algorithm,
                    env_id=benchmark,
                    seed=seed,
                    total_timesteps=args.total_timesteps,
                    script_path=args.script_path,
                    timeout=args.timeout,
                )
                
                all_results.append(result)
                
                # Print result
                if result["status"] == "success":
                    return_str = f" (return: {result.get('final_return', 'N/A')})" if "final_return" in result else ""
                    print(f"  ✓ Success in {result['training_time']:.1f}s{return_str}")
                else:
                    error_msg = f": {result['error']}" if result.get("error") else ""
                    print(f"  ✗ {result['status'].upper()}{error_msg}")
                
                if pbar:
                    pbar.update(1)
    
    if pbar:
        pbar.close()
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Print summary
    print_summary_table(all_results)
    
    # Print failures summary
    failures = [r for r in all_results if r["status"] != "success"]
    if failures:
        print(f"\n⚠ {len(failures)} experiment(s) failed:")
        for f in failures:
            print(f"  - {f['algorithm']} on {f['benchmark']} (seed {f['seed']}): {f['status']}")
            if f.get("error"):
                print(f"    Error: {f['error']}")


if __name__ == "__main__":
    main()

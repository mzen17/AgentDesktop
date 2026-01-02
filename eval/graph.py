import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

def load_results(filepath="eval/results.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_and_plot(results):
    agents = list(results.keys())
    
    # Extract data
    distances = {agent: [] for agent in agents}
    successes = {agent: [] for agent in agents}
    
    for agent in agents:
        # Ensure we align data by test_id if possible, but list order should match 
        # since benchmark runs sequentially in the same loop.
        for r in results[agent]:
            if r.get("distance") is not None:
                distances[agent].append(r["distance"])
                successes[agent].append(1 if r["success"] else 0)
                
    # Calculate stats
    means_dist = [np.mean(distances[a]) for a in agents]
    sems_dist = [stats.sem(distances[a]) for a in agents]
    
    means_succ = [np.mean(successes[a]) for a in agents]
    sems_succ = [stats.sem(successes[a]) for a in agents]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distance Plot
    x = np.arange(len(agents))
    width = 0.35
    
    rects1 = ax1.bar(x, means_dist, width, yerr=sems_dist, capsize=5, label='Distance', color='skyblue')
    ax1.set_ylabel('Pixels')
    ax1.set_title('Average Distance to Target (Lower is Better)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Success Rate Plot
    rects2 = ax2.bar(x, means_succ, width, yerr=sems_succ, capsize=5, color='lightgreen', label='Success Rate')
    ax2.set_ylabel('Rate (0-1)')
    ax2.set_title('Success Rate (Higher is Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(agents)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("eval/benchmark_results.png")
    print("Graph saved to eval/benchmark_results.png")
    
    # Statistical Analysis (Paired t-test if 2 agents)
    if len(agents) == 2:
        a1, a2 = agents[0], agents[1]
        print(f"\nStatistical Analysis ({a1} vs {a2}):")
        
        # Distance
        # Check if we have enough data
        if len(distances[a1]) > 1 and len(distances[a2]) > 1:
            t_stat, p_val = stats.ttest_rel(distances[a1], distances[a2])
            print(f"Distance: t-stat={t_stat:.4f}, p-value={p_val:.4f}")
            if p_val < 0.05:
                print("  -> Significant difference in distance.")
            else:
                print("  -> No significant difference in distance.")
        
        # Success
        if len(successes[a1]) > 1 and len(successes[a2]) > 1:
            t_stat, p_val = stats.ttest_rel(successes[a1], successes[a2])
            print(f"Success: t-stat={t_stat:.4f}, p-value={p_val:.4f}")
            if p_val < 0.05:
                print("  -> Significant difference in success rate.")
            else:
                print("  -> No significant difference in success rate.")

if __name__ == "__main__":
    if not os.path.exists("eval/results.json"):
        print("No results found. Run benchmark.py first.")
    else:
        results = load_results()
        analyze_and_plot(results)

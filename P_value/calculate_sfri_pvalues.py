import os
import json
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Constants
RESULTS_FILE = "results/enhanced_comparison.json"
P_VALUE_FILE = "P_value/sfri_pvalues.json"
P_VALUE_PLOT = "P_value/sfri_pvalue_significance.png"

def load_results():
    """Load comparison results from JSON file"""
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def extract_sfri_data(results):
    """Extract SFRI data from episode metrics for each agent"""
    sfri_data = {}
    
    for agent_type in results.keys():
        if "metrics" in results[agent_type] and "episode_metrics" in results[agent_type]["metrics"]:
            episode_metrics = results[agent_type]["metrics"]["episode_metrics"]
            sfri_scores = [em["sfri"] for em in episode_metrics if "sfri" in em]
            
            if sfri_scores:
                sfri_data[agent_type] = sfri_scores
    
    return sfri_data

def calculate_pvalues(sfri_data):
    """Calculate p-values for all pairs of agents using t-test"""
    p_values = {}
    agent_types = list(sfri_data.keys())
    
    print("SFRI Score Statistics:")
    for agent in agent_types:
        scores = sfri_data[agent]
        print(f"  {agent}: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}, min={np.min(scores):.2f}, max={np.max(scores):.2f}")
    
    print("\nP-Value Analysis (two-tailed t-test):")
    for i in range(len(agent_types)):
        for j in range(i+1, len(agent_types)):
            agent1 = agent_types[i]
            agent2 = agent_types[j]
            
            # Perform two-tailed t-test
            t_stat, p_val = stats.ttest_ind(sfri_data[agent1], sfri_data[agent2], equal_var=False)
            
            # Create a key for this comparison
            comparison_key = f"{agent1}_vs_{agent2}"
            
            # Store results
            p_values[comparison_key] = {
                "p_value": float(p_val),  # Convert numpy types to Python native types
                "t_statistic": float(t_stat),
                "significant_0.05": bool(p_val < 0.05),  # Explicitly convert to Python bool
                "significant_0.01": bool(p_val < 0.01),
                "significant_0.001": bool(p_val < 0.001),
                "mean_diff": float(np.mean(sfri_data[agent1]) - np.mean(sfri_data[agent2])),
                "better_agent": agent1 if np.mean(sfri_data[agent1]) > np.mean(sfri_data[agent2]) else agent2
            }
            
            # Print interpretation
            significance = "not significant"
            if p_val < 0.001:
                significance = "highly significant (p<0.001)"
            elif p_val < 0.01:
                significance = "very significant (p<0.01)"
            elif p_val < 0.05:
                significance = "significant (p<0.05)"
                
            better = p_values[comparison_key]["better_agent"]
            mean_diff = abs(p_values[comparison_key]["mean_diff"])
            
            print(f"  {agent1} vs {agent2}: p={p_val:.6f} - {significance}")
            print(f"    {better} performs better by {mean_diff:.2f} SFRI points")
    
    return p_values

def visualize_pvalue_results(sfri_data, p_values):
    """Create visualization of SFRI scores and significance"""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: SFRI boxplot
    agent_types = list(sfri_data.keys())
    boxplot_data = [sfri_data[agent] for agent in agent_types]
    
    ax1.boxplot(boxplot_data, labels=agent_types, patch_artist=True)
    ax1.set_title('SFRI Score Distribution by Agent Type')
    ax1.set_ylabel('SFRI Score (0-100)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Significance matrix
    significance_matrix = np.zeros((len(agent_types), len(agent_types)))
    
    for i in range(len(agent_types)):
        for j in range(i+1, len(agent_types)):
            agent1 = agent_types[i]
            agent2 = agent_types[j]
            comparison_key = f"{agent1}_vs_{agent2}"
            
            if comparison_key in p_values:
                p_val = p_values[comparison_key]["p_value"]
                
                # Encode significance level in the matrix
                if p_val < 0.001:
                    significance_matrix[i, j] = 3  # *** (p<0.001)
                    significance_matrix[j, i] = 3
                elif p_val < 0.01:
                    significance_matrix[i, j] = 2  # ** (p<0.01)
                    significance_matrix[j, i] = 2
                elif p_val < 0.05:
                    significance_matrix[i, j] = 1  # * (p<0.05)
                    significance_matrix[j, i] = 1
                else:
                    significance_matrix[i, j] = 0  # ns (not significant)
                    significance_matrix[j, i] = 0
    
    # Create heatmap
    im = ax2.imshow(significance_matrix, cmap='YlOrRd')
    
    # Add significance labels
    for i in range(len(agent_types)):
        for j in range(len(agent_types)):
            if i != j:
                if significance_matrix[i, j] == 3:
                    text = "***"
                elif significance_matrix[i, j] == 2:
                    text = "**"
                elif significance_matrix[i, j] == 1:
                    text = "*"
                else:
                    text = "ns"
                ax2.text(j, i, text, ha="center", va="center", color="black")
            else:
                ax2.text(j, i, "-", ha="center", va="center", color="black")
    
    # Add labels
    ax2.set_title('Statistical Significance Between Agents')
    ax2.set_xticks(np.arange(len(agent_types)))
    ax2.set_yticks(np.arange(len(agent_types)))
    ax2.set_xticklabels(agent_types)
    ax2.set_yticklabels(agent_types)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['Not Significant', 'p<0.05 (*)', 'p<0.01 (**)', 'p<0.001 (***)'])
    
    plt.tight_layout()
    plt.savefig(P_VALUE_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved p-value significance visualization to {P_VALUE_PLOT}")

def save_pvalue_results(p_values):
    """Save p-value results to JSON file"""
    try:
        with open(P_VALUE_FILE, 'w') as f:
            json.dump(p_values, f, indent=2)
        print(f"P-value results saved to {P_VALUE_FILE}")
    except Exception as e:
        print(f"Error saving p-value results: {e}")

def main():
    """Calculate and save p-values for SFRI scores"""
    print("Calculating p-values for SFRI scores...")
    
    # Load results
    results = load_results()
    if not results:
        print("Failed to load results. Exiting.")
        return
    
    # Extract SFRI data
    sfri_data = extract_sfri_data(results)
    if not sfri_data:
        print("No SFRI data found in results. Exiting.")
        return
    
    # Calculate p-values
    p_values = calculate_pvalues(sfri_data)
    
    # Save results
    save_pvalue_results(p_values)
    
    # Create visualization
    visualize_pvalue_results(sfri_data, p_values)
    
    print("P-value calculation complete.")

if __name__ == "__main__":
    main() 
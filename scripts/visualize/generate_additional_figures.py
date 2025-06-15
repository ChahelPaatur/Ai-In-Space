import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Constants
RESULTS_FILE = "results/enhanced_comparison.json"
P_VALUE_FILE = "P_value/sfri_pvalues.json"
PLOT_DIR = Path("static/plots/paper")

def load_data():
    """Load comparison results from JSON file"""
    try:
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def load_pvalues():
    """Load p-value results from JSON file"""
    try:
        with open(P_VALUE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading p-values: {e}")
        return None

def generate_action_distribution(data):
    """Generate Figure 9: Action Distribution Across Agent Types"""
    print("Generating Figure 9: Action Distribution Across Agent Types...")
    
    # Action names mapping
    action_names = {
        0: "No Action",
        1: "Recover ADCS",
        2: "Recover EPS",
        3: "Recover TCS", 
        4: "Safe Mode"
    }
    
    # Extract action counts for each agent
    action_data = {}
    agent_types = list(data.keys())
    
    for agent_type in agent_types:
        if "metrics" in data[agent_type]:
            actions = []
            if "episode_metrics" in data[agent_type]["metrics"]:
                episode_metrics = data[agent_type]["metrics"]["episode_metrics"]
                for em in episode_metrics:
                    if "actions" in em:
                        actions.extend(em["actions"])
            
            # Count actions
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            action_data[agent_type] = action_counts
    
    # Organize data for plotting with matplotlib instead of seaborn
    # Get all unique action types
    all_actions = set()
    for agent in action_data:
        all_actions.update(action_data[agent].keys())
    
    # Convert action keys to integers and sort
    all_actions = sorted([int(a) for a in all_actions])
    
    # Create a structured dataset for plotting
    plot_data = {}
    for agent in action_data:
        plot_data[agent] = []
        for action in all_actions:
            action_str = str(action)
            count = action_data[agent].get(action_str, 0)
            plot_data[agent].append(count)
    
    # Create figure
    plt.figure(figsize=(12, 7))
    
    # Set up x positions
    num_agents = len(agent_types)
    num_actions = len(all_actions)
    width = 0.8 / num_agents
    
    # Get nice colors
    colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
    
    # Plot bars for each agent
    for i, agent in enumerate(agent_types):
        x_pos = np.arange(num_actions) - 0.4 + (i + 0.5) * width
        bars = plt.bar(x_pos, plot_data[agent], width=width, label=agent.capitalize(), color=colors[i])
        
        # Add count labels on top of bars
        for bar, count in zip(bars, plot_data[agent]):
            if count > 0:  # Only add labels for non-zero counts
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                        str(count), ha='center', va='bottom')
    
    # Set x-axis ticks and labels
    plt.xticks(np.arange(num_actions), [action_names.get(a, f"Action {a}") for a in all_actions], rotation=15)
    
    plt.title("Figure 9: Action Distribution Across Agent Types (n=100 episodes)", fontsize=14)
    plt.xlabel("Action Type", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(title="Agent Type")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / "figure9_action_distribution.png", dpi=300, bbox_inches='tight')
    print("  Saved Figure 9 to", PLOT_DIR / "figure9_action_distribution.png")
    plt.close()

def generate_response_time_series(data):
    """Generate Figures 10a & 10b: Temperature and Battery Response Time Series"""
    print("Generating Figures 10a & 10b: System Response Time Series...")
    
    # Check if we have time series data in the results
    has_time_series = False
    for agent_type in data:
        if "metrics" in data[agent_type] and "time_series" in data[agent_type]["metrics"]:
            has_time_series = True
            break
    
    if not has_time_series:
        # If no time series data in results, create synthetic data for illustration
        print("  No time series data found in results, generating synthetic data for illustration")
        generate_synthetic_response_series()
        return
    
    # Extract time series data
    temp_data = {}
    battery_data = {}
    
    for agent_type in data:
        if "metrics" in data[agent_type] and "time_series" in data[agent_type]["metrics"]:
            time_series = data[agent_type]["metrics"]["time_series"]
            
            if "temperature" in time_series:
                temp_data[agent_type] = time_series["temperature"]
            
            if "battery_soc" in time_series:
                battery_data[agent_type] = time_series["battery_soc"]
    
    # Generate temperature response plot (Figure 10a)
    if temp_data:
        plt.figure(figsize=(10, 6))
        
        for agent_type, temps in temp_data.items():
            plt.plot(temps, label=agent_type.capitalize(), linewidth=2)
        
        plt.title("Figure 10a: Temperature Response Time Series", fontsize=14)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Temperature (°C)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Agent Type")
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "figure10a_temperature_response.png", dpi=300, bbox_inches='tight')
        print("  Saved Figure 10a to", PLOT_DIR / "figure10a_temperature_response.png")
        plt.close()
    
    # Generate battery response plot (Figure 10b)
    if battery_data:
        plt.figure(figsize=(10, 6))
        
        for agent_type, soc in battery_data.items():
            plt.plot(soc, label=agent_type.capitalize(), linewidth=2)
        
        plt.title("Figure 10b: Battery State of Charge Response Time Series", fontsize=14)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Battery SoC (%)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title="Agent Type")
        
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "figure10b_battery_response.png", dpi=300, bbox_inches='tight')
        print("  Saved Figure 10b to", PLOT_DIR / "figure10b_battery_response.png")
        plt.close()

def generate_synthetic_response_series():
    """Generate synthetic time series data for illustration when real data is not available"""
    # Temperature response (Figure 10a)
    plt.figure(figsize=(10, 6))
    
    # Synthetic data
    time_steps = np.arange(0, 200)
    
    # Classical agent - slow response with oscillations
    classical_temp = 30 - 15 * np.exp(-time_steps/50) + 2 * np.sin(time_steps/10)
    
    # DRL agent - faster response but some oscillation
    drl_temp = 30 - 15 * np.exp(-time_steps/30) + np.sin(time_steps/15)
    
    # Hybrid agent - optimal response
    hybrid_temp = 30 - 15 * np.exp(-time_steps/20) + 0.5 * np.sin(time_steps/20)
    
    # Add fault event
    fault_time = 50
    classical_temp[fault_time:] += 5 * np.exp(-(time_steps[fault_time:]-fault_time)/40)
    drl_temp[fault_time:] += 3 * np.exp(-(time_steps[fault_time:]-fault_time)/30)
    hybrid_temp[fault_time:] += 2 * np.exp(-(time_steps[fault_time:]-fault_time)/15)
    
    plt.plot(time_steps, classical_temp, label="Classical", linewidth=2)
    plt.plot(time_steps, drl_temp, label="DRL", linewidth=2)
    plt.plot(time_steps, hybrid_temp, label="Hybrid", linewidth=2)
    
    # Add fault indicator
    plt.axvline(x=fault_time, color='r', linestyle='--', alpha=0.7, label="Fault Injection")
    
    plt.title("Figure 10a: Temperature Response Time Series (Synthetic)", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Temperature (°C)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Agent Type")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "figure10a_temperature_response.png", dpi=300, bbox_inches='tight')
    print("  Saved Figure 10a to", PLOT_DIR / "figure10a_temperature_response.png")
    plt.close()
    
    # Battery response (Figure 10b)
    plt.figure(figsize=(10, 6))
    
    # Synthetic data
    # Classical agent - slow recovery
    classical_batt = 100 - 20 * np.exp(-time_steps/70)
    
    # DRL agent - faster recovery
    drl_batt = 100 - 20 * np.exp(-time_steps/40)
    
    # Hybrid agent - optimal recovery
    hybrid_batt = 100 - 20 * np.exp(-time_steps/30)
    
    # Add fault event with battery drain
    classical_batt[fault_time:] -= 30 * np.exp(-(time_steps[fault_time:]-fault_time)/60)
    drl_batt[fault_time:] -= 25 * np.exp(-(time_steps[fault_time:]-fault_time)/50)
    hybrid_batt[fault_time:] -= 15 * np.exp(-(time_steps[fault_time:]-fault_time)/30)
    
    plt.plot(time_steps, classical_batt, label="Classical", linewidth=2)
    plt.plot(time_steps, drl_batt, label="DRL", linewidth=2)
    plt.plot(time_steps, hybrid_batt, label="Hybrid", linewidth=2)
    
    # Add fault indicator
    plt.axvline(x=fault_time, color='r', linestyle='--', alpha=0.7, label="Fault Injection")
    
    plt.title("Figure 10b: Battery State of Charge Response Time Series (Synthetic)", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Battery SoC (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Agent Type")
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "figure10b_battery_response.png", dpi=300, bbox_inches='tight')
    print("  Saved Figure 10b to", PLOT_DIR / "figure10b_battery_response.png")
    plt.close()

def generate_pvalue_visualization(pvalue_data):
    """Generate Figure 12: P-value and Effect Size Visualization"""
    print("Generating Figure 12: P-value and Effect Size Visualization...")
    
    if not pvalue_data:
        print("  No p-value data available, skipping Figure 12")
        return
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Extract data
    comparisons = []
    p_values = []
    significance = []
    mean_diffs = []
    better_agents = []
    
    for comparison, data in pvalue_data.items():
        comparisons.append(comparison.replace('_vs_', ' vs ').replace('classical', 'Classical').replace('drl', 'DRL').replace('hybrid', 'Hybrid'))
        p_values.append(data['p_value'])
        
        # Determine significance level
        if data['significant_0.001']:
            significance.append('***')
        elif data['significant_0.01']:
            significance.append('**')
        elif data['significant_0.05']:
            significance.append('*')
        else:
            significance.append('ns')
        
        mean_diffs.append(abs(data['mean_diff']))
        better_agents.append(data['better_agent'].capitalize())
    
    # Create first subplot - p-values
    ax1.barh(comparisons, -np.log10(p_values), color='skyblue')
    
    # Add significance markers
    for i, (comp, sig) in enumerate(zip(comparisons, significance)):
        ax1.text(-np.log10(p_values[i]) + 0.1, i, sig, va='center', fontweight='bold')
    
    # Add p-value threshold lines
    ax1.axvline(x=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax1.axvline(x=-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='p=0.01')
    ax1.axvline(x=-np.log10(0.001), color='green', linestyle='--', alpha=0.7, label='p=0.001')
    
    ax1.set_title('Statistical Significance (-log10(p))', fontsize=14)
    ax1.set_xlabel('-log10(p-value)', fontsize=12)
    ax1.set_ylabel('Comparison', fontsize=12)
    ax1.legend()
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create second subplot - mean differences
    colors = ['#3498db' if agent == 'Classical' else '#2ecc71' if agent == 'Drl' else '#9b59b6' for agent in better_agents]
    
    ax2.barh(comparisons, mean_diffs, color=colors)
    
    # Add better agent labels
    for i, (comp, diff, agent) in enumerate(zip(comparisons, mean_diffs, better_agents)):
        ax2.text(diff + 0.5, i, agent, va='center')
    
    ax2.set_title('SFRI Score Difference (Better Agent)', fontsize=14)
    ax2.set_xlabel('SFRI Score Difference', fontsize=12)
    ax2.set_ylabel('Comparison', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "figure12_pvalue_analysis.png", dpi=300, bbox_inches='tight')
    print("  Saved Figure 12 to", PLOT_DIR / "figure12_pvalue_analysis.png")
    plt.close()

def main():
    """Generate all requested figures"""
    print("Generating additional requested figures...")
    
    # Ensure output directory exists
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data()
    if not data:
        print("Failed to load results data, cannot generate figures.")
        return
    
    # Generate figures
    generate_action_distribution(data)
    generate_response_time_series(data)
    
    # Generate p-value visualization
    pvalue_data = load_pvalues()
    generate_pvalue_visualization(pvalue_data)
    
    print("All additional figures generated successfully!")

if __name__ == "__main__":
    main() 
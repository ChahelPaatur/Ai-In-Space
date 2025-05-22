import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.express as px

# Add the project root to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Constants for visualization
RESULTS_FILE = "results/enhanced_comparison.json"  # Summary results file
LOGS_DIR = "logs"  # Directory containing detailed episode logs
PLOT_DIR = "static/plots"  # Directory to save generated plots
ROLLING_WINDOW = 10  # Window for smoothing in time series plots

# Configuration
CLASSICAL_RESULTS_FILE = "results/comparison_summary.json"
HYBRID_RESULTS_FILE = "results/hybrid_summary.json"
OUTPUT_DIR = "static/plots" 

def load_all_logs():
    """Load all episode logs from the logs directory."""
    all_data = {
        'classical': [],
        'drl': [],
        'hybrid': []
    }
    
    # Load each agent type's logs
    for agent_type in all_data.keys():
        log_files = glob.glob(os.path.join(LOGS_DIR, f"{agent_type}_episode_*.json"))
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    episode_data = json.load(f)
                    episode_num = int(os.path.basename(log_file).split('_')[-1].split('.')[0])
                    
                    # Add metadata to each step
                    for step in episode_data:
                        step['agent_type'] = agent_type
                        step['episode'] = episode_num
                    
                    all_data[agent_type].extend(episode_data)
            except Exception as e:
                print(f"Error loading {log_file}: {e}")
    
    return all_data

def calculate_metrics_from_logs(logs_data):
    """Calculate MTTR, MTTD, false positives, and other metrics from log data."""
    metrics = {}
    
    for agent_type, logs in logs_data.items():
        # Group by episode
        episode_groups = {}
        for step in logs:
            episode = step['episode']
            if episode not in episode_groups:
                episode_groups[episode] = []
            episode_groups[episode].append(step)
        
        # Calculate metrics for each episode
        mttr_values = []
        mttd_values = []
        detection_rates = []
        recovery_rates = []
        false_positives = []
        
        for episode, steps in episode_groups.items():
            # Get fault episodes
            fault_episodes = []
            current_episode = None
            
            for step_idx, step in enumerate(steps):
                faults = step.get('persistent_faults', [])
                action = step.get('action')
                
                # Track fault episodes
                if faults and current_episode is None:
                    current_episode = {
                        'start_step': step_idx,
                        'faults': faults.copy(),
                        'detection_step': None,
                        'recovery_step': None,
                        'actions_taken': []
                    }
                elif faults and current_episode is not None:
                    # Update fault list
                    for fault in faults:
                        if fault not in current_episode['faults']:
                            current_episode['faults'].append(fault)
                    
                    # Check for recovery action
                    if action in [1, 2, 3]:  # RecoverEPS, RecoverADCS, RecoverTCS
                        if current_episode['detection_step'] is None:
                            current_episode['detection_step'] = step_idx
                    
                    current_episode['actions_taken'].append(action)
                    
                    # Check if faults are resolved in next step
                    next_step = steps[step_idx + 1] if step_idx + 1 < len(steps) else None
                    if next_step and not next_step.get('persistent_faults', []) and current_episode['recovery_step'] is None:
                        current_episode['recovery_step'] = step_idx + 1
                        fault_episodes.append(current_episode)
                        current_episode = None
                
                elif not faults and current_episode is not None:
                    # End of fault episode
                    if current_episode['recovery_step'] is None:
                        current_episode['recovery_step'] = step_idx
                    fault_episodes.append(current_episode)
                    current_episode = None
            
            # Handle any incomplete episode
            if current_episode is not None:
                current_episode['recovery_step'] = len(steps)
                fault_episodes.append(current_episode)
            
            # Calculate episode metrics
            ep_ttd = []
            ep_ttr = []
            ep_fp = 0  # False positives
            
            for fault_ep in fault_episodes:
                # Time to detect
                if fault_ep['detection_step'] is not None:
                    ttd = fault_ep['detection_step'] - fault_ep['start_step']
                    ep_ttd.append(ttd)
                
                # Time to recover
                if fault_ep['recovery_step'] is not None:
                    ttr = fault_ep['recovery_step'] - fault_ep['start_step']
                    ep_ttr.append(ttr)
            
            # Count recovery actions when no fault present (false positives)
            for step_idx, step in enumerate(steps):
                if step.get('action') in [1, 2, 3] and not step.get('persistent_faults', []):
                    # Check if it's part of a recent recovery (grace period)
                    is_recent_recovery = False
                    for fault_ep in fault_episodes:
                        if fault_ep['recovery_step'] is not None:
                            # Allow 5 steps after recovery as grace period
                            if 0 <= step_idx - fault_ep['recovery_step'] <= 5:
                                is_recent_recovery = True
                                break
                    
                    if not is_recent_recovery:
                        ep_fp += 1
            
            # Store episode metrics
            mttd = np.mean(ep_ttd) if ep_ttd else float('inf')
            mttr = np.mean(ep_ttr) if ep_ttr else float('inf')
            detection_rate = len(ep_ttd) / len(fault_episodes) if fault_episodes else 1.0
            recovery_rate = len(ep_ttr) / len(fault_episodes) if fault_episodes else 1.0
            
            if mttd != float('inf'):
                mttd_values.append(mttd)
            if mttr != float('inf'):
                mttr_values.append(mttr)
            
            detection_rates.append(detection_rate)
            recovery_rates.append(recovery_rate)
            false_positives.append(ep_fp)
        
        # Calculate agent-level metrics
        metrics[agent_type] = {
            'mttd': np.mean(mttd_values) if mttd_values else float('inf'),
            'mttd_std': np.std(mttd_values) if len(mttd_values) > 1 else 0,
            'mttr': np.mean(mttr_values) if mttr_values else float('inf'),
            'mttr_std': np.std(mttr_values) if len(mttr_values) > 1 else 0,
            'detection_rate': np.mean(detection_rates),
            'recovery_rate': np.mean(recovery_rates),
            'false_positives_avg': np.mean(false_positives),
            'false_positives_total': sum(false_positives),
            'mttd_values': mttd_values,
            'mttr_values': mttr_values,
            'detection_rates': detection_rates,
            'recovery_rates': recovery_rates,
            'false_positives': false_positives
        }
    
    return metrics

def plot_mttr_mttd_comparison(metrics, output_dir=OUTPUT_DIR):
    """Create box plots comparing MTTR and MTTD across agents."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    data = {
        'Agent': [],
        'Metric': [],
        'Value': []
    }
    
    for agent, agent_metrics in metrics.items():
        # Add MTTD values
        for value in agent_metrics['mttd_values']:
            data['Agent'].append(agent.capitalize())
            data['Metric'].append('MTTD')
            data['Value'].append(value)
        
        # Add MTTR values
        for value in agent_metrics['mttr_values']:
            data['Agent'].append(agent.capitalize())
            data['Metric'].append('MTTR')
            data['Value'].append(value)
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.boxplot(x='Agent', y='Value', hue='Metric', data=df, palette='Set3')
    ax.set_title('MTTD and MTTR Comparison Across Agents', fontsize=16)
    ax.set_ylabel('Steps', fontsize=14)
    ax.set_xlabel('Agent Type', fontsize=14)
    
    # Add mean values as text
    for agent_type in metrics.keys():
        mttd = metrics[agent_type]['mttd']
        mttr = metrics[agent_type]['mttr']
        
        if mttd != float('inf'):
            ax.text(list(metrics.keys()).index(agent_type) - 0.2, 
                    mttd + 0.5, 
                    f'μ={mttd:.1f}', 
                    ha='center', 
                    va='bottom')
        
        if mttr != float('inf'):
            ax.text(list(metrics.keys()).index(agent_type) + 0.2, 
                    mttr + 0.5, 
                    f'μ={mttr:.1f}', 
                    ha='center', 
                    va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mttr_mttd_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Saved MTTR/MTTD comparison plot to {os.path.join(output_dir, 'mttr_mttd_comparison.png')}")

def plot_detection_recovery_rates(metrics, output_dir=OUTPUT_DIR):
    """Create bar plots of detection and recovery rates."""
    os.makedirs(output_dir, exist_ok=True)
    
    agents = list(metrics.keys())
    detection_rates = [metrics[a]['detection_rate'] * 100 for a in agents]
    recovery_rates = [metrics[a]['recovery_rate'] * 100 for a in agents]
    
    x = np.arange(len(agents))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, detection_rates, width, label='Detection Rate')
    rects2 = ax.bar(x + width/2, recovery_rates, width, label='Recovery Rate')
    
    ax.set_title('Fault Detection and Recovery Rates', fontsize=16)
    ax.set_ylabel('Rate (%)', fontsize=14)
    ax.set_xlabel('Agent Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in agents])
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_recovery_rates.png'), dpi=300)
    plt.close()
    
    print(f"Saved detection/recovery rates plot to {os.path.join(output_dir, 'detection_recovery_rates.png')}")

def plot_false_positives(metrics, output_dir=OUTPUT_DIR):
    """Create bar plot of false positive recovery actions."""
    os.makedirs(output_dir, exist_ok=True)
    
    agents = list(metrics.keys())
    fp_avg = [metrics[a]['false_positives_avg'] for a in agents]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(agents)), fp_avg, color='salmon')
    
    plt.title('Average False Positive Recovery Actions per Episode', fontsize=16)
    plt.ylabel('False Positives', fontsize=14)
    plt.xlabel('Agent Type', fontsize=14)
    plt.xticks(range(len(agents)), [a.capitalize() for a in agents])
    
    # Add value labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.1, 
                 f'{fp_avg[i]:.2f}', 
                 ha='center', 
                 va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'false_positives.png'), dpi=300)
    plt.close()
    
    print(f"Saved false positives plot to {os.path.join(output_dir, 'false_positives.png')}")

def plot_hybrid_decision_distribution(logs_data, output_dir=OUTPUT_DIR):
    """Create a pie chart showing hybrid agent decision source distribution."""
    os.makedirs(output_dir, exist_ok=True)
    
    if 'hybrid' not in logs_data:
        print("No hybrid agent data found for decision distribution plot.")
        return
    
    # Count decision sources
    decision_counts = {}
    for step in logs_data['hybrid']:
        if 'decision_source' in step:
            source = step['decision_source']
            decision_counts[source] = decision_counts.get(source, 0) + 1
    
    if not decision_counts:
        print("No decision source data found in hybrid agent logs.")
        return
    
    # Create pie chart
    labels = list(decision_counts.keys())
    sizes = list(decision_counts.values())
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Hybrid Agent Decision Source Distribution', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_decision_distribution.png'), dpi=300)
    plt.close()
    
    print(f"Saved hybrid decision distribution plot to {os.path.join(output_dir, 'hybrid_decision_distribution.png')}")

def create_agent_architecture_diagrams(output_dir=OUTPUT_DIR):
    """Create diagrams of agent architectures (DRL and Hybrid)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. DRL Agent Architecture
    plt.figure(figsize=(12, 8))
    
    # Define the components
    components = {
        'Input Layer': (0.5, 0.9, 'lightblue'),
        'Hidden Layer 1': (0.5, 0.7, 'lightgreen'),
        'Hidden Layer 2': (0.5, 0.5, 'lightgreen'),
        'Actor Head': (0.3, 0.3, 'salmon'),
        'Critic Head': (0.7, 0.3, 'orange'),
        'Action Distribution': (0.3, 0.1, 'yellow'),
        'Value Estimate': (0.7, 0.1, 'yellow')
    }
    
    # Draw the components
    for name, (x, y, color) in components.items():
        plt.scatter(x, y, s=3000, color=color, alpha=0.6, edgecolors='black')
        plt.text(x, y, name, ha='center', va='center', fontsize=12)
    
    # Draw the connections
    plt.arrow(0.5, 0.9, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.7, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.5, -0.15, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.5, 0.15, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.3, 0.3, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.7, 0.3, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    plt.title('DRL Agent Architecture (Actor-Critic Network)', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drl_architecture.png'), dpi=300)
    plt.close()
    
    # 2. Hybrid Agent Architecture
    plt.figure(figsize=(14, 10))
    
    # Define the components for hybrid architecture
    hybrid_components = {
        'Spacecraft Telemetry': (0.5, 0.95, 'lightblue'),
        'Rule-Based System': (0.25, 0.75, 'lightgreen'),
        'DRL System': (0.75, 0.75, 'salmon'),
        'Safety Critical?': (0.25, 0.55, 'yellow'),
        'DRL Confidence > Threshold?': (0.75, 0.55, 'yellow'),
        'Decision Arbitration': (0.5, 0.35, 'orange'),
        'Final Action': (0.5, 0.15, 'lightblue')
    }
    
    # Draw the components
    for name, (x, y, color) in hybrid_components.items():
        plt.scatter(x, y, s=5000, color=color, alpha=0.6, edgecolors='black')
        plt.text(x, y, name, ha='center', va='center', fontsize=12)
    
    # Draw the connections
    plt.arrow(0.5, 0.95, -0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.95, 0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.25, 0.75, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.75, 0.75, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.25, 0.55, 0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.75, 0.55, -0.2, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.35, 0, -0.1, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Add labels for decision flows
    plt.text(0.15, 0.65, 'Yes: Safety Override', fontsize=10, ha='center')
    plt.text(0.35, 0.65, 'No', fontsize=10, ha='center')
    plt.text(0.65, 0.65, 'Yes: Use DRL', fontsize=10, ha='center')
    plt.text(0.85, 0.65, 'No: Use Rule-based', fontsize=10, ha='center')
    
    plt.title('Hybrid FDIR Agent Architecture', fontsize=16)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hybrid_architecture.png'), dpi=300)
    plt.close()
    
    print(f"Saved agent architecture diagrams to {output_dir}")

def plot_reward_comparison_all_agents(output_dir=OUTPUT_DIR):
    """Create a comprehensive reward comparison plot including hybrid agent."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_data = {}
    
    try:
        if os.path.exists(CLASSICAL_RESULTS_FILE):
            with open(CLASSICAL_RESULTS_FILE, 'r') as f:
                classical_drl_results = json.load(f)
                for agent_type, data in classical_drl_results.items():
                    results_data[agent_type] = data
    except Exception as e:
        print(f"Error loading classical/DRL results: {e}")
    
    try:
        if os.path.exists(HYBRID_RESULTS_FILE):
            with open(HYBRID_RESULTS_FILE, 'r') as f:
                hybrid_results = json.load(f)
                for agent_type, data in hybrid_results.items():
                    results_data[agent_type] = data
    except Exception as e:
        print(f"Error loading hybrid results: {e}")
    
    if not results_data:
        print("No results data found for reward comparison.")
        return
    
    # Prepare data for plotting
    data = {
        'Agent': [],
        'Reward': []
    }
    
    for agent_type, agent_data in results_data.items():
        rewards = agent_data.get('rewards', [])
        for reward in rewards:
            data['Agent'].append(agent_type.capitalize())
            data['Reward'].append(reward)
    
    df = pd.DataFrame(data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    ax = sns.boxplot(x='Agent', y='Reward', data=df, palette='viridis')
    ax.set_title('Total Reward Comparison Across All Agents', fontsize=16)
    ax.set_ylabel('Total Episode Reward', fontsize=14)
    ax.set_xlabel('Agent Type', fontsize=14)
    
    # Add mean values as text
    for agent_type in results_data.keys():
        rewards = results_data[agent_type].get('rewards', [])
        if rewards:
            mean_reward = np.mean(rewards)
            idx = list(results_data.keys()).index(agent_type)
            ax.text(idx, mean_reward, f'μ={mean_reward:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_agents_reward_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Saved all-agent reward comparison plot to {os.path.join(output_dir, 'all_agents_reward_comparison.png')}")

def main():
    """Main function to run all visualization tasks."""
    print("--- Advanced Visualization Starting ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load log data
    print("Loading logs...")
    logs_data = load_all_logs()
    
    # Check if we have any log data
    if not any(logs_data.values()):
        print("No log data found. Please run the agents first.")
        return
    
    # Calculate metrics from logs
    print("Calculating metrics...")
    metrics = calculate_metrics_from_logs(logs_data)
    
    # Create architecture diagrams
    print("Creating architecture diagrams...")
    create_agent_architecture_diagrams()
    
    # Create comparison plots
    print("Creating comparison plots...")
    plot_mttr_mttd_comparison(metrics)
    plot_detection_recovery_rates(metrics)
    plot_false_positives(metrics)
    
    # Create hybrid-specific plots if data available
    if 'hybrid' in logs_data and logs_data['hybrid']:
        print("Creating hybrid agent plots...")
        plot_hybrid_decision_distribution(logs_data)
    
    # Create reward comparison including all agents
    print("Creating reward comparison plot...")
    plot_reward_comparison_all_agents()
    
    print("--- Advanced Visualization Complete ---")

if __name__ == "__main__":
    main() 
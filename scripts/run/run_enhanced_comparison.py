import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.spacecraft_env import SpacecraftEnv
from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent
from src.hybrid_agent import EnhancedHybridFDIRAgent
# from src.agents.safety_compliant_hybrid_agent import SafetyCompliantHybridAgent
from src.metrics import FDIRMetrics

# --- Simulation Configuration ---
# Fault scenarios based on historical spacecraft failures and mission data and plausible fault studies:
# - Mars Climate Orbiter/Polar Lander power system failures
# - Hubble Space Telescope gyro and battery issues  
# - ISS thermal control and power management anomalies
# - Kepler reaction wheel failures and Dawn spacecraft ADCS problems
# - ESA OPS-SAT telemetry patterns and NASA spacecraft specifications

# Paper reference: Section 3.5 "Evaluation Methodology" - Running 100 episodes per agent type
# "All three agent types (Rule-based, DRL, and Hybrid) were evaluated over 100 episodes each"
NUM_EPISODES = 100           # Increased number of episodes for statistical significance
MAX_STEPS_PER_EPISODE = 200  # Maximum steps per episode before truncation
FAULT_PROBABILITY = 0.02     # Per-step probability of injecting a new persistent fault
RENDER_MODE = None           # Set to 'human' for visual rendering, None for faster runs
DRL_MODEL_PATH = "ppo_agent.pth"  # Path to the default trained DRL agent model
LONG_DRL_MODEL_PATH = "ppo_agent_long.pth"  # Path to the longer-trained model (if available)
RESULTS_FILE = "results/enhanced_comparison.json"  # Output file for enhanced metrics
LOGS_DIR = "logs"            # Directory to save detailed step-by-step JSON logs
SAVE_DETAILED_LOGS = True    # Set to False to disable detailed logging
CONFIDENCE_THRESHOLD = 0.85  # Increased threshold for hybrid agent DRL confidence (was 0.7)

def run_agent(agent_type, agent_instance, env, results_data, metrics_tracker=None):
    """
    Run evaluation episodes for any agent type with enhanced metrics tracking.
    
    # Paper reference: Section 3.5 "Evaluation Methodology" - This function implements the
    # comparative evaluation of different agent types using identical environment configurations.
    
    Args:
        agent_type: String identifier for the agent ('classical', 'drl', 'hybrid', etc.)
        agent_instance: The agent object with a get_action method
        env: SpacecraftEnv instance
        results_data: Dictionary to store results
        metrics_tracker: Optional FDIRMetrics instance for advanced metrics
    """
    print(f"--- Running Simulation with {agent_type.capitalize()} Agent ---")
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Create a new metrics tracker if not provided
    if metrics_tracker is None:
        metrics_tracker = FDIRMetrics()
    
    # Results tracking
    episode_rewards = []
    episode_steps = []
    episode_metrics = []
    
    # Paper reference: Section 3.5 - Running 100 episodes per agent type to ensure statistical validity
    for episode in range(NUM_EPISODES):
        print(f"\n--- {agent_type.capitalize()}: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset()
        
        # CRITICAL: Reset agent state for new episode (especially hybrid agent)
        if hasattr(agent_instance, 'reset'):
            agent_instance.reset()
        
        current_episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_log = []  # Log for the current episode
        
        while not terminated and not truncated:
            # Get action from agent (handle different agent interfaces)
            # Paper reference: Section 3.4 - Hybrid agent uses confidence-based arbitration mechanism
            if agent_type == 'hybrid':
                action, decision_info = agent_instance.get_action(observation, info)
                decision_source = decision_info['decision_source']
            elif agent_type.startswith('drl'):
                # DRL agent - use enhanced_get_action for our conservative thresholds
                drl_result = agent_instance.enhanced_get_action(observation, info)
                if len(drl_result) == 4:  # If it returns (action, log_prob, value, diagnostics)
                    action, log_prob, value, diagnostics = drl_result
                    decision_source = diagnostics.get('source', agent_type.capitalize())
                else:  # If it returns (action, log_prob, value)
                    action, log_prob, value = drl_result
                    decision_source = agent_type.capitalize()
            else:
                action = agent_instance.get_action(observation, info)
                decision_source = agent_type.capitalize()
            
            # Get raw observation for logging
            current_raw_obs = info.get("raw_observation", observation)
            
            # Step the environment
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            current_episode_reward += reward
            step += 1
            
            # Log detailed step data
            # Paper reference: Section 4.2 - The data collected here enables the action repertoire
            # and dynamic response analysis described in the paper
            if SAVE_DETAILED_LOGS:
                step_data = {
                    'step': step,
                    'observation': current_raw_obs.tolist() if isinstance(current_raw_obs, np.ndarray) else current_raw_obs,
                    'action': action,
                    'reward': reward,
                    'mode': info.get('mode', 'Unknown'),
                    'is_sunlit': info.get('is_sunlit', True),
                    'subsystem_statuses': info.get('subsystem_statuses', {}),
                    'persistent_faults': info.get('active_faults_persistent', []),
                    'intermittent_faults': info.get('active_faults_intermittent', []),
                    'decision_source': decision_source,
                    'terminated': terminated,
                    'truncated': truncated
                }
                
                # Add hybrid-specific decision info if available
                # Paper reference: Section 4.2 "Hybrid Decision Distribution" - This data enables 
                # the analysis shown in Figure 6 of the paper
                if agent_type == 'hybrid':
                    step_data.update({
                        'drl_confidence': decision_info['drl_confidence'],
                        'rule_action': decision_info['rule_action'],
                        'drl_action': decision_info['drl_action']
                    })
                
                episode_log.append(step_data)
            
            # Prepare for next iteration
            observation = next_observation
            info = next_info
            
            # Check for episode end conditions
            if terminated:
                print(f"{agent_type.capitalize()} Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"{agent_type.capitalize()} Episode finished after {step} steps (Truncated)")
        
        # Calculate enhanced metrics for this episode
        # Paper reference: Section 3.5 "Metrics Framework" - Calculates the specialized FDIR metrics
        # including MTTR, detection rate, recovery rate, false positives, and SFRI
        # Note: MTTD is still calculated for analysis but no longer used in SFRI calculation
        # SFRI weights: Detection (35%), False Positive (30%), Recovery (25%), Stability (10%)
        if episode_log:
            episode_metrics_data = metrics_tracker.process_episode_log(episode_log)
            episode_metrics.append(episode_metrics_data)
            
            print(f"{agent_type.capitalize()} Episode {episode + 1} metrics:")
            print(f"  MTTD: {episode_metrics_data['mttd']:.2f} steps")
            print(f"  MTTR: {episode_metrics_data['mttr']:.2f} steps")
            print(f"  Detection Rate: {episode_metrics_data['detection_rate']*100:.1f}%")
            print(f"  Recovery Rate: {episode_metrics_data['recovery_rate']*100:.1f}%")
            print(f"  False Positives: {episode_metrics_data['false_positives']}")
            print(f"  SFRI Score: {episode_metrics_data['sfri']:.1f}/70")
        
        # End of Episode summary
        end_time = time.time()
        print(f"{agent_type.capitalize()} Episode {episode + 1} finished. Reward: {current_episode_reward:.2f}")
        print(f"  Final Status: {info.get('subsystem_statuses', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")
        
        # Store results
        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)
        
        # Save detailed log
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"{agent_type}_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")
    
    # Calculate aggregate metrics
    aggregate_metrics = metrics_tracker.get_aggregate_metrics()
    
    # Store results
    # Paper reference: Section 4.1 - This data is used to generate the aggregate performance
    # metrics presented in Figures 1-5 of the paper
    results_data[agent_type] = {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'avg_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'metrics': {
            'mttd': float(aggregate_metrics['mttd']),
            'mttr': float(aggregate_metrics['mttr']),
            'detection_rate': float(aggregate_metrics['detection_rate']),
            'recovery_rate': float(aggregate_metrics['recovery_rate']),
            'false_positives': int(aggregate_metrics['false_positives']),
            'stability_impact': float(aggregate_metrics['stability_impact']),
            'sfri': float(aggregate_metrics['sfri']),
            'episode_metrics': episode_metrics
        }
    }
    
    # Print summary
    # Paper reference: Section 4.1 - These values directly correspond to the metrics
    # reported in "Aggregate Performance Metrics" section of the paper
    print(f"\n--- Simulation Summary ({agent_type.capitalize()} Agent) ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Avg Reward: {results_data[agent_type]['avg_reward']:.2f} ± {results_data[agent_type]['std_reward']:.2f}")
    print(f"Avg MTTD: {aggregate_metrics['mttd']:.2f} steps")
    print(f"Avg MTTR: {aggregate_metrics['mttr']:.2f} steps")
    print(f"Detection Rate: {aggregate_metrics['detection_rate']*100:.1f}%")
    print(f"Recovery Rate: {aggregate_metrics['recovery_rate']*100:.1f}%")
    print(f"False Positives: {aggregate_metrics['false_positives']}")
    print(f"Stability Impact: {aggregate_metrics['stability_impact']:.4f}")
    print(f"SFRI Score: {aggregate_metrics['sfri']:.1f}/70")
    
    return results_data

def run_classical_agent(results_data):
    """Run Rule-Based FDIR agent with enhanced metrics."""
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=False  # Raw observations for rule-based
    )
    
    agent = RuleBasedFDIR()
    metrics = FDIRMetrics()
    
    run_agent('classical', agent, env, results_data, metrics)
    env.close()

def run_drl_agent(results_data, model_path=DRL_MODEL_PATH, agent_type='drl'):
    """Run DRL agent with enhanced metrics."""
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: DRL model file not found at {model_path}. Skipping DRL agent run.")
        return results_data
    
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=True  # Normalized observations for DRL
    )
    
    # Create and load agent
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(obs_size, action_size)
    
    try:
        agent.load_model(model_path)
        agent.network.eval()
        print(f"Successfully loaded DRL model from {model_path}")
    except Exception as e:
        print(f"Error loading DRL model: {e}. Skipping DRL agent run.")
        env.close()
        return results_data
    
    metrics = FDIRMetrics()
    run_agent(agent_type, agent, env, results_data, metrics)
    env.close()
    
    return results_data

def run_hybrid_agent(results_data):
    """Run Hybrid FDIR agent with enhanced metrics."""
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=True  # Normalized for DRL component
    )
    
    # Check if DRL model exists
    if not os.path.exists(DRL_MODEL_PATH):
        print(f"Error: DRL model file not found at {DRL_MODEL_PATH}. Cannot create hybrid agent.")
        env.close()
        return results_data
    
    # Create hybrid agent
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    try:
        from src.hybrid_agent import EnhancedHybridFDIRAgent as HybridFDIRAgent
        
        # Updated initialization with BALANCED settings for optimal SFRI
        agent = HybridFDIRAgent(
            obs_size=15,
            action_size=9,
            confidence_threshold=0.18,  # BALANCED for actual DRL confidence levels!
            external_data_support=True,
            config={'confirmation_steps': 2}  # Balanced temporal validation
        )
    except ImportError:
        print("Error: EnhancedHybridFDIRAgent not found. Please ensure src.hybrid_agent is available.")
        env.close()
        return results_data
    
    metrics = FDIRMetrics()
    run_agent('hybrid', agent, env, results_data, metrics)
    env.close()
    
    return results_data

def run_long_trained_drl(results_data):
    """Run the longer-trained DRL agent if available."""
    if not os.path.exists(LONG_DRL_MODEL_PATH):
        print(f"Long-trained DRL model not found at {LONG_DRL_MODEL_PATH}. Skipping.")
        return results_data
    
    return run_drl_agent(results_data, LONG_DRL_MODEL_PATH, 'drl_long')

def run_enhanced_comparison():
    """
    Run the complete enhanced comparison with all agent types.
    
    # Paper reference: Section 3.5 "Comparative Evaluation" - This is the main function that 
    # implements the evaluation methodology described in the paper, running the three agent types
    # (Rule-based, DRL, and Hybrid) under identical environment configurations.
    """
    # Dictionary to hold results from all agent runs
    results = {}
    
    # Create results directory
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Run all agents
    # Paper reference: Section 3.5 - Evaluating all three agent types
    run_classical_agent(results)
    run_drl_agent(results)
    run_hybrid_agent(results)
    run_long_trained_drl(results)
    
    # Save results
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Generate comparison table
    # Paper reference: Section 4.1 - This table summarizes the key metrics that are
    # analyzed in detail in the "Aggregate Performance Metrics" section
    print("\n=== AGENT COMPARISON SUMMARY ===")
    print(f"{'Agent':<15} {'Reward':<15} {'MTTD':<10} {'MTTR':<10} {'SFRI':<10}")
    print("-" * 60)
    
    for agent_type, data in results.items():
        metrics = data.get('metrics', {})
        reward = f"{data['avg_reward']:.2f} ± {data['std_reward']:.2f}"
        mttd = f"{metrics.get('mttd', 'N/A'):.2f}"
        mttr = f"{metrics.get('mttr', 'N/A'):.2f}"
        sfri = f"{metrics.get('sfri', 'N/A'):.1f}"
        
        print(f"{agent_type.capitalize():<15} {reward:<15} {mttd:<10} {mttr:<10} {sfri:<10}")
    
    return results

def create_comparison_plots(results):
    """Create comparison plots and save them to the static/plots/paper directory."""
    plot_dir = Path("static/plots/paper")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot colors and labels
    colors = {
        "classical": "blue",
        "drl": "orange",
        "hybrid": "green",
        "safety_hybrid": "red"
    }
    
    agent_labels = {
        "classical": "Classical",
        "drl": "DRL",
        "hybrid": "Hybrid",
        "safety_hybrid": "Safety-Compliant Hybrid"
    }
    
    # 1. Reward Plot (Figure 1)
    plt.figure(figsize=(10, 8))
    for agent_type in results.keys():
        if not results[agent_type]["rewards"]:
            continue
        rewards = results[agent_type]["rewards"]
        mean_reward = np.mean(rewards)
        plt.boxplot([rewards], positions=[list(results.keys()).index(agent_type)],
                  widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor=colors[agent_type], alpha=0.7))
        plt.scatter([list(results.keys()).index(agent_type)] * len(rewards), 
                  rewards, color=colors[agent_type], alpha=0.4, s=20)
        plt.scatter([list(results.keys()).index(agent_type)], [mean_reward], 
                  color='black', marker='*', s=100, zorder=3)
        
    plt.xticks(range(len(results)), [agent_labels[a] for a in results.keys()])
    plt.ylabel("Episode Reward")
    plt.title("Figure 1: Total Episode Reward (n=100)")
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / "figure1_reward_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MTTD & MTTR Plot (Figure 2)
    plt.figure(figsize=(10, 8))
    
    # Gather data
    mttd_values = []
    mttr_values = []
    agent_types = []
    
    for agent_type in results.keys():
        if agent_type in results and "metrics" in results[agent_type]:
            metrics = results[agent_type]["metrics"]
            if "mttd" in metrics and "mttr" in metrics:
                mttd_values.append(metrics["mttd"])
                mttr_values.append(metrics["mttr"])
                agent_types.append(agent_labels[agent_type])
    
    x = np.arange(len(agent_types))
    width = 0.35
    
    plt.bar(x - width/2, mttd_values, width, label='MTTD', color='skyblue')
    plt.bar(x + width/2, mttr_values, width, label='MTTR', color='lightcoral')
    
    plt.xlabel('Agent Type')
    plt.ylabel('Time Steps')
    plt.title('Figure 2: Detection and Recovery Time (n=100)')
    plt.xticks(x, agent_types)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add values as text above each bar
    for i, v in enumerate(mttd_values):
        plt.text(i - width/2, v + 2, f"{v:.1f}", ha='center', fontsize=9)
    for i, v in enumerate(mttr_values):
        plt.text(i + width/2, v + 2, f"{v:.1f}", ha='center', fontsize=9)
        
    plt.savefig(plot_dir / "figure2_mttr_mttd_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. False Positives Plot (Figure 3)
    plt.figure(figsize=(10, 8))
    
    # Gather data
    fp_values = []
    agent_types = []
    
    for agent_type in results.keys():
        if agent_type in results and "metrics" in results[agent_type]:
            metrics = results[agent_type]["metrics"]
            if "false_positives" in metrics:
                fp_values.append(metrics["false_positives"])
                agent_types.append(agent_labels[agent_type])
    
    plt.bar(agent_types, fp_values, color=[colors[a.lower()] for a in agent_types])
    plt.xlabel('Agent Type')
    plt.ylabel('Number of False Positives')
    plt.title('Figure 3: False Positive Recovery Actions (n=100)')
    plt.grid(True, alpha=0.3)
    
    # Add values as text above each bar
    for i, v in enumerate(fp_values):
        plt.text(i, v + max(fp_values) * 0.02, f"{v}", ha='center')
        
    plt.savefig(plot_dir / "figure3_false_positives.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. SFRI Comparison (Figure 4)
    plt.figure(figsize=(10, 8))
    
    # Gather data
    sfri_values = []
    agent_types = []
    
    for agent_type in results.keys():
        if agent_type in results and "metrics" in results[agent_type]:
            metrics = results[agent_type]["metrics"]
            if "sfri" in metrics:
                sfri_values.append(metrics["sfri"])
                agent_types.append(agent_labels[agent_type])
    
    plt.bar(agent_types, sfri_values, color=[colors[a.lower()] for a in agent_types])
    plt.xlabel('Agent Type')
    plt.ylabel('SFRI Score')
    plt.title('Figure 4: SFRI Score Comparison (n=100)')
    plt.ylim(0, 70)  # Scale to max possible score
    plt.grid(True, alpha=0.3)
    
    # Add values as text above each bar
    for i, v in enumerate(sfri_values):
        plt.text(i, v + 1, f"{v:.1f}", ha='center')
        
    plt.savefig(plot_dir / "figure4_sfri_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Detection & Recovery Rate (Figure 5)
    plt.figure(figsize=(10, 8))
    
    # Gather data
    detection_rates = []
    recovery_rates = []
    agent_types = []
    
    for agent_type in results.keys():
        if agent_type in results and "metrics" in results[agent_type]:
            metrics = results[agent_type]["metrics"]
            if "detection_rate" in metrics and "recovery_rate" in metrics:
                detection_rates.append(metrics["detection_rate"] * 100)  # Convert to percentage
                recovery_rates.append(metrics["recovery_rate"] * 100)    # Convert to percentage
                agent_types.append(agent_labels[agent_type])
    
    x = np.arange(len(agent_types))
    width = 0.35
    
    plt.bar(x - width/2, detection_rates, width, label='Detection Rate', color='skyblue')
    plt.bar(x + width/2, recovery_rates, width, label='Recovery Rate', color='lightgreen')
    
    plt.xlabel('Agent Type')
    plt.ylabel('Rate (%)')
    plt.title('Figure 5: Fault Detection and Recovery Rates (n=100)')
    plt.xticks(x, agent_types)
    plt.legend()
    plt.ylim(0, 105)  # Scale to percentage
    plt.grid(True, alpha=0.3)
    
    # Add values as text above each bar
    for i, v in enumerate(detection_rates):
        plt.text(i - width/2, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
    for i, v in enumerate(recovery_rates):
        plt.text(i + width/2, v + 2, f"{v:.1f}%", ha='center', fontsize=9)
        
    plt.savefig(plot_dir / "figure5_detection_recovery_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All comparison plots saved to {plot_dir}")

if __name__ == "__main__":
    run_enhanced_comparison() 
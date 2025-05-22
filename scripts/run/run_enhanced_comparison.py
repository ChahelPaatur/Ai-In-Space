import numpy as np
import time
import torch
import json
import os
import glob
import sys

# Add the project root to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.spacecraft_env import SpacecraftEnv
from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent
from src.hybrid_agent import HybridFDIRAgent
from src.metrics import calculate_mttr_mttd, calculate_sfri

# --- Simulation Configuration ---
NUM_EPISODES = 100           # Increased number of episodes for statistical significance
MAX_STEPS_PER_EPISODE = 200  # Maximum steps per episode before truncation
FAULT_PROBABILITY = 0.02     # Per-step probability of injecting a new persistent fault
RENDER_MODE = None           # Set to 'human' for visual rendering, None for faster runs
DRL_MODEL_PATH = "ppo_agent.pth"  # Path to the default trained DRL agent model
LONG_DRL_MODEL_PATH = "ppo_agent_long.pth"  # Path to the longer-trained model (if available)
RESULTS_FILE = "results/enhanced_comparison.json"  # Output file for enhanced metrics
LOGS_DIR = "logs"            # Directory to save detailed step-by-step JSON logs
SAVE_DETAILED_LOGS = True    # Set to False to disable detailed logging
CONFIDENCE_THRESHOLD = 0.7   # Threshold for hybrid agent DRL confidence

def run_agent(agent_type, agent_instance, env, results_data, metrics_tracker=None):
    """
    Run evaluation episodes for any agent type with enhanced metrics tracking.
    
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
    
    for episode in range(NUM_EPISODES):
        print(f"\n--- {agent_type.capitalize()}: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset()
        current_episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_log = []  # Log for the current episode
        
        while not terminated and not truncated:
            # Get action from agent (handle different agent interfaces)
            if agent_type == 'hybrid':
                action, decision_info = agent_instance.get_action(observation, info)
                decision_source = decision_info['decision_source']
            elif agent_type.startswith('drl'):
                # DRL agent doesn't use info parameter
                action, log_prob, value = agent_instance.get_action(observation)
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
        if episode_log:
            episode_metrics_data = metrics_tracker.process_episode_log(episode_log)
            episode_metrics.append(episode_metrics_data)
            
            print(f"{agent_type.capitalize()} Episode {episode + 1} metrics:")
            print(f"  MTTD: {episode_metrics_data['mttd']:.2f} steps")
            print(f"  MTTR: {episode_metrics_data['mttr']:.2f} steps")
            print(f"  Detection Rate: {episode_metrics_data['detection_rate']*100:.1f}%")
            print(f"  Recovery Rate: {episode_metrics_data['recovery_rate']*100:.1f}%")
            print(f"  False Positives: {episode_metrics_data['false_positives']}")
            print(f"  SFRI Score: {episode_metrics_data['sfri']:.1f}/100")
        
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
    print(f"\n--- Simulation Summary ({agent_type.capitalize()} Agent) ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Avg Reward: {results_data[agent_type]['avg_reward']:.2f} ± {results_data[agent_type]['std_reward']:.2f}")
    print(f"Avg MTTD: {aggregate_metrics['mttd']:.2f} steps")
    print(f"Avg MTTR: {aggregate_metrics['mttr']:.2f} steps")
    print(f"Detection Rate: {aggregate_metrics['detection_rate']*100:.1f}%")
    print(f"Recovery Rate: {aggregate_metrics['recovery_rate']*100:.1f}%")
    print(f"False Positives: {aggregate_metrics['false_positives']}")
    print(f"Stability Impact: {aggregate_metrics['stability_impact']:.4f}")
    print(f"SFRI Score: {aggregate_metrics['sfri']:.1f}/100")
    
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
    agent = HybridFDIRAgent(
        obs_size=obs_size,
        action_size=action_size,
        drl_model_path=DRL_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
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
    """Run the complete enhanced comparison with all agent types."""
    # Dictionary to hold results from all agent runs
    results = {}
    
    # Create results directory
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Run all agents
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

if __name__ == "__main__":
    run_enhanced_comparison() 
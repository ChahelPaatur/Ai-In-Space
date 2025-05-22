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
from src.hybrid_agent import HybridFDIRAgent

# --- Simulation Configuration ---
NUM_EPISODES = 100           # Increased number of episodes for statistical validity
MAX_STEPS_PER_EPISODE = 200  # Maximum steps per episode before truncation
FAULT_PROBABILITY = 0.02     # Per-step probability of injecting a new persistent fault
RENDER_MODE = None           # Set to 'human' for visual rendering, None for faster runs
DRL_MODEL_PATH = "ppo_agent.pth" # Path to the saved trained DRL agent model
RESULTS_FILE = "results/hybrid_summary.json" # Output file for summary statistics
LOGS_DIR = "logs"            # Directory to save detailed step-by-step JSON logs
SAVE_DETAILED_LOGS = True    # Set to False to disable detailed logging
CONFIDENCE_THRESHOLD = 0.7   # Threshold for DRL confidence to override rules

def calculate_mttr_mttd(episode_log, fault_field='persistent_faults', detection_actions=[1, 2, 3]):
    """
    Calculate Mean Time To Detect (MTTD) and Mean Time To Recover (MTTR) metrics.
    
    Args:
        episode_log: List of step dictionaries containing fault and action data
        fault_field: Field name in the log containing fault information
        detection_actions: List of action indices considered as recovery actions
        
    Returns:
        dict: Dictionary containing MTTD and MTTR metrics
    """
    fault_episodes = []  # List to track each fault episode
    current_episode = None
    
    # Process the log to identify fault episodes
    for step_idx, step_data in enumerate(episode_log):
        faults = step_data.get(fault_field, [])
        action = step_data.get('action')
        
        # Case 1: New fault detected, no current tracking
        if faults and current_episode is None:
            current_episode = {
                'start_step': step_idx,
                'faults': faults.copy(),
                'detection_step': None,
                'recovery_step': None,
                'actions_taken': []
            }
            
        # Case 2: Ongoing fault episode
        elif faults and current_episode is not None:
            # Update fault list if needed
            for fault in faults:
                if fault not in current_episode['faults']:
                    current_episode['faults'].append(fault)
            
            # Check if this is a recovery action
            if action in detection_actions and current_episode['detection_step'] is None:
                current_episode['detection_step'] = step_idx
                
            # Record all actions for analysis
            current_episode['actions_taken'].append(action)
            
            # If we previously detected and now faults are resolved, mark recovery
            next_step = episode_log[step_idx + 1] if step_idx + 1 < len(episode_log) else None
            if next_step and not next_step.get(fault_field, []) and current_episode['recovery_step'] is None:
                current_episode['recovery_step'] = step_idx + 1
                fault_episodes.append(current_episode)
                current_episode = None
                
        # Case 3: No faults, but we were tracking an episode
        elif not faults and current_episode is not None:
            # End of fault without recovery action
            if current_episode['recovery_step'] is None:
                current_episode['recovery_step'] = step_idx
            fault_episodes.append(current_episode)
            current_episode = None
            
    # Handle any incomplete episode at the end
    if current_episode is not None:
        current_episode['recovery_step'] = len(episode_log)  # Mark as unrecovered
        fault_episodes.append(current_episode)
    
    # Calculate metrics
    ttd_values = []  # Time To Detect
    ttr_values = []  # Time To Recover
    
    for episode in fault_episodes:
        # Time to detect (if detected)
        if episode['detection_step'] is not None:
            ttd = episode['detection_step'] - episode['start_step']
            ttd_values.append(ttd)
            
        # Time to recover (if recovered)
        if episode['recovery_step'] is not None and episode['start_step'] is not None:
            ttr = episode['recovery_step'] - episode['start_step']
            ttr_values.append(ttr)
    
    # Calculate means
    mttd = np.mean(ttd_values) if ttd_values else float('inf')
    mttr = np.mean(ttr_values) if ttr_values else float('inf')
    
    return {
        'mttd': mttd,
        'mttr': mttr,
        'detection_rate': len(ttd_values) / len(fault_episodes) if fault_episodes else 0,
        'recovery_rate': len(ttr_values) / len(fault_episodes) if fault_episodes else 0,
        'fault_episodes': len(fault_episodes),
        'ttd_values': ttd_values,
        'ttr_values': ttr_values
    }

def run_hybrid_agent():
    """Runs evaluation episodes using the hybrid FDIR agent."""
    print("--- Running Simulation with Hybrid FDIR Agent ---")
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Instantiate Environment - Use raw observations for rule component
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=True  # Normalize for DRL component
    )
    
    # Get observation and action space sizes
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create the hybrid agent
    agent = HybridFDIRAgent(
        obs_size=obs_size,
        action_size=action_size,
        drl_model_path=DRL_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Results tracking
    episode_rewards = []
    episode_steps = []
    decision_sources = []
    episode_mttd = []
    episode_mttr = []
    episode_detection_rates = []
    episode_recovery_rates = []
    
    for episode in range(NUM_EPISODES):
        print(f"\n--- Hybrid: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset()
        current_episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_log = []  # Log for the current episode
        episode_decisions = []  # Track decision sources for this episode
        
        while not terminated and not truncated:
            # Get action from hybrid agent (with decision info)
            action, decision_info = agent.get_action(observation, info)
            episode_decisions.append(decision_info['decision_source'])
            
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
                    'decision_source': decision_info['decision_source'],
                    'drl_confidence': decision_info['drl_confidence'],
                    'rule_action': decision_info['rule_action'],
                    'drl_action': decision_info['drl_action'],
                    'terminated': terminated,
                    'truncated': truncated
                }
                episode_log.append(step_data)
            
            # Prepare for next iteration
            observation = next_observation
            info = next_info
            
            # Check for episode end conditions
            if terminated:
                print(f"Hybrid Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"Hybrid Episode finished after {step} steps (Truncated - Max steps reached)")
        
        # Calculate MTTR and MTTD metrics for this episode
        metrics = calculate_mttr_mttd(episode_log)
        
        # End of Episode
        end_time = time.time()
        print(f"Hybrid Episode {episode + 1} finished. Reward: {current_episode_reward:.2f}")
        print(f"  Final Status: {info.get('subsystem_statuses', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")
        print(f"  MTTD: {metrics['mttd']:.2f}, MTTR: {metrics['mttr']:.2f}")
        print(f"  Decision sources: Rule: {episode_decisions.count('Rule-Based')}, " 
              f"DRL: {episode_decisions.count('DRL')}, " 
              f"Override: {episode_decisions.count('Rule (Safety Override)')}, "
              f"DRL Confident: {episode_decisions.count('DRL (High Confidence)')}")
        
        # Store results
        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)
        decision_sources.extend(episode_decisions)
        episode_mttd.append(metrics['mttd'])
        episode_mttr.append(metrics['mttr'])
        episode_detection_rates.append(metrics['detection_rate'])
        episode_recovery_rates.append(metrics['recovery_rate'])
        
        # Save detailed log
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"hybrid_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")
    
    env.close()
    
    # Aggregate results
    results = {
        'hybrid': {
            'rewards': episode_rewards,
            'steps': episode_steps,
            'avg_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'decision_distribution': {
                'Rule-Based': decision_sources.count('Rule-Based'),
                'DRL': decision_sources.count('DRL'),
                'Rule (Safety Override)': decision_sources.count('Rule (Safety Override)'),
                'DRL (High Confidence)': decision_sources.count('DRL (High Confidence)')
            },
            'mttd': float(np.mean([x for x in episode_mttd if x != float('inf')])),
            'mttr': float(np.mean([x for x in episode_mttr if x != float('inf')])),
            'detection_rate': float(np.mean(episode_detection_rates)),
            'recovery_rate': float(np.mean(episode_recovery_rates))
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n--- Simulation Summary (Hybrid Agent) ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Avg Reward: {results['hybrid']['avg_reward']:.2f} ± {results['hybrid']['std_reward']:.2f}")
    print(f"Avg MTTD: {results['hybrid']['mttd']:.2f}")
    print(f"Avg MTTR: {results['hybrid']['mttr']:.2f}")
    print(f"Detection Rate: {results['hybrid']['detection_rate'] * 100:.2f}%")
    print(f"Recovery Rate: {results['hybrid']['recovery_rate'] * 100:.2f}%")
    print(f"Decision Distribution: {results['hybrid']['decision_distribution']}")
    
    return results

if __name__ == "__main__":
    run_hybrid_agent() 
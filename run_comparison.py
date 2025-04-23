import numpy as np
import time
import torch # Needed for DRL agent
import json # For saving results
import os # For checking file existence
import glob # To find log files later

# Use absolute imports assuming 'src' is in the Python path or run from project root
from src.spacecraft_env import SpacecraftEnv
from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent # Import the DRL agent

# --- Simulation Parameters ---
NUM_EPISODES = 5
MAX_STEPS_PER_EPISODE = 200 # Shorten for quick testing
FAULT_PROBABILITY = 0.02 # Increase probability for more frequent faults during testing
RENDER_MODE = None # Disable rendering during logging runs for speed
DRL_MODEL_PATH = "ppo_agent.pth" # Path to the saved DRL agent model
RESULTS_FILE = "results.json" # Summary results
LOGS_DIR = "logs" # Directory for detailed episode logs
SAVE_DETAILED_LOGS = True # Flag to enable/disable detailed logging


def run_classical_agent(results_data):
    """Runs the simulation with the classical FDIR agent."""
    print("--- Running Simulation with Classical FDIR Agent ---")
    os.makedirs(LOGS_DIR, exist_ok=True) # Ensure log directory exists

    # Instantiate Environment and Agent
    env = SpacecraftEnv(
        render_mode=None, # Control rendering manually
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=False # Get raw obs directly for logging/classical agent
    )
    agent = RuleBasedFDIR()

    episode_rewards = []
    episode_steps = []
    # final_fault_statuses = [] # Not saving detailed fault status for simplicity now

    for episode in range(NUM_EPISODES):
        print(f"\n--- Classical: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset()
        current_episode_reward = 0 # Renamed to avoid clash with list name
        terminated = False
        truncated = False
        step = 0
        episode_log = [] # Initialize log for this episode

        while not terminated and not truncated:
            action = agent.get_action(observation, info)
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            current_episode_reward += reward
            step += 1

            # Log step data if enabled
            if SAVE_DETAILED_LOGS:
                step_data = {
                    'step': step,
                    'observation': observation.tolist(), # Current obs before action
                    'action': action,
                    'reward': reward,
                    'mode': info.get('mode', 'Unknown'),
                    'is_sunlit': info.get('is_sunlit', True),
                    'subsystem_statuses': info.get('subsystem_statuses', {}),
                    'persistent_faults': info.get('active_faults_persistent', []),
                    'intermittent_faults': info.get('active_faults_intermittent', []),
                    'terminated': terminated,
                    'truncated': truncated
                }
                episode_log.append(step_data)

            # Update observation and info for the next loop iteration
            observation = next_observation
            info = next_info

            if terminated:
                print(f"Classical Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"Classical Episode finished after {step} steps (Truncated - Max steps reached)")

        end_time = time.time()
        print(f"Classical Episode {episode + 1} finished. Reward: {current_episode_reward:.2f}")
        print(f"  Final Fault Status: {info.get('subsystem_statuses', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")

        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)
        # final_fault_statuses.append(info['subsystem_faults'])

        # Save detailed log for this episode
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"classical_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
                # print(f"Saved detailed log to {log_filename}")
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")

    env.close()

    # Store results
    results_data['classical'] = {
        'rewards': episode_rewards,
        'steps': episode_steps
    }

    print("\n--- Simulation Summary (Classical Agent) ---")
    print(f"Number of Episodes: {NUM_EPISODES}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} +/- {np.std(episode_steps):.2f}")


def run_drl_agent(results_data, model_path=DRL_MODEL_PATH):
    """Runs the simulation with a pre-trained DRL agent."""
    print(f"\n--- Running Simulation with DRL Agent ({model_path}) ---")
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Instantiate Environment
    env = SpacecraftEnv(
        render_mode=None, # Control rendering manually
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=True 
    )

    # Instantiate Agent and Load Model
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(obs_size, action_size)
    model_loaded = False
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            agent.network.eval() # Set network to evaluation mode for inference
            model_loaded = True
        except Exception as e:
            print(f"Error loading DRL model: {e}. Skipping DRL agent run.")
            env.close()
            return
    else:
        print(f"DRL model file not found at {model_path}. Skipping DRL agent run.")
        env.close()
        return

    episode_rewards = []
    episode_steps = []
    # final_fault_statuses = []

    for episode in range(NUM_EPISODES):
        print(f"\n--- DRL: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset() # Observation is normalized here
        current_episode_reward = 0 # Renamed
        terminated = False
        truncated = False
        step = 0
        episode_log = [] # Initialize log for this episode

        while not terminated and not truncated:
            # Get action from DRL agent using potentially normalized observation
            action, _, _ = agent.get_action(observation)
            # Get current raw observation *before* stepping (from info dict)
            current_raw_obs = info.get("raw_observation", observation if not env.normalize_obs else [])
            
            # Step the environment
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            current_episode_reward += reward
            step += 1

            # Log step data if enabled (using raw obs)
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
                    'terminated': terminated,
                    'truncated': truncated
                }
                episode_log.append(step_data)

            # Update observation and info for the next loop iteration
            observation = next_observation
            info = next_info

            if terminated:
                print(f"DRL Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"DRL Episode finished after {step} steps (Truncated - Max steps reached)")

        end_time = time.time()
        print(f"DRL Episode {episode + 1} finished.")
        print(f"  Total Reward: {current_episode_reward:.2f}")
        print(f"  Final Fault Status: {info.get('subsystem_faults', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")

        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)
        # final_fault_statuses.append(info['subsystem_faults'])

        # Save detailed log for this episode
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"drl_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
                # print(f"Saved detailed log to {log_filename}")
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")

    env.close()

    # Store results
    results_data['drl'] = {
        'rewards': episode_rewards,
        'steps': episode_steps
    }

    print(f"\n--- Simulation Summary (DRL Agent - {model_path}) ---")
    print(f"Number of Episodes: {NUM_EPISODES}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average Steps: {np.mean(episode_steps):.2f} +/- {np.std(episode_steps):.2f}")

if __name__ == "__main__":
    results = {}
    run_classical_agent(results)
    run_drl_agent(results) # Pass the results dict

    # Save results to JSON file
    if results: # Only save if we have some results
        try:
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nSummary results saved to {RESULTS_FILE}")
        except Exception as e:
            print(f"\nError saving summary results to {RESULTS_FILE}: {e}")
    else:
        print("\nNo summary results generated to save.")

    print(f"\nDetailed episode logs saved in: {LOGS_DIR}" if SAVE_DETAILED_LOGS else "") 
import numpy as np
import time
import torch # Needed for loading/running the DRL agent
import json
import os
import glob

from src.spacecraft_env import SpacecraftEnv
from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent

# --- Simulation Configuration ---
NUM_EPISODES = 5            # Number of simulation runs for each agent
MAX_STEPS_PER_EPISODE = 200 # Maximum steps per episode before truncation
FAULT_PROBABILITY = 0.02    # Per-step probability of injecting a new persistent fault
RENDER_MODE = None          # Set to 'human' for visual rendering, None for faster runs
DRL_MODEL_PATH = "ppo_agent.pth" # Path to the saved trained DRL agent model
RESULTS_FILE = "results/comparison_summary.json" # Output file for summary statistics (rewards, steps)
LOGS_DIR = "logs"             # Directory to save detailed step-by-step JSON logs
SAVE_DETAILED_LOGS = True     # Set to False to disable detailed logging


def run_classical_agent(results_data):
    """Runs evaluation episodes using the classical rule-based FDIR agent."""
    print("--- Running Simulation with Classical FDIR Agent ---")
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Instantiate Environment - Note: Use raw observations for classical agent
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=False
    )
    agent = RuleBasedFDIR()

    episode_rewards = []
    episode_steps = []

    for episode in range(NUM_EPISODES):
        print(f"\n--- Classical: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        observation, info = env.reset()
        current_episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_log = [] # Log for the current episode

        while not terminated and not truncated:
            # Classical agent determines action based on current observation/info
            action = agent.get_action(observation, info)
            next_observation, reward, terminated, truncated, next_info = env.step(action)

            current_episode_reward += reward
            step += 1

            # Log detailed step data if enabled
            if SAVE_DETAILED_LOGS:
                step_data = {
                    'step': step,
                    'observation': observation.tolist(), # Log observation *before* action
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

            # Prepare for next iteration
            observation = next_observation
            info = next_info

            # --- Check for episode end conditions --- #
            if terminated:
                print(f"Classical Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"Classical Episode finished after {step} steps (Truncated - Max steps reached)")

        # --- End of Episode --- #
        end_time = time.time()
        print(f"Classical Episode {episode + 1} finished. Reward: {current_episode_reward:.2f}")
        print(f"  Final Status: {info.get('subsystem_statuses', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")

        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)

        # Save detailed log for this episode
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"classical_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")

    env.close()

    # Store aggregated results for this agent
    results_data['classical'] = {
        'rewards': episode_rewards,
        'steps': episode_steps
    }

    print("\n--- Simulation Summary (Classical Agent) ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")


def run_drl_agent(results_data, model_path=DRL_MODEL_PATH):
    """Runs evaluation episodes using the pre-trained DRL agent."""
    print(f"\n--- Running Simulation with DRL Agent ({model_path}) ---")
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Check if the DRL model file exists before proceeding
    if not os.path.exists(model_path):
        print(f"Error: DRL model file not found at {model_path}. Skipping DRL agent run.")
        return

    # Instantiate Environment - Note: Use normalized observations for DRL agent
    env = SpacecraftEnv(
        render_mode=RENDER_MODE,
        fault_probability=FAULT_PROBABILITY,
        max_steps=MAX_STEPS_PER_EPISODE,
        normalize_obs=True
    )

    # Instantiate Agent and Load Model
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PPOAgent(obs_size, action_size)

    try:
        agent.load_model(model_path)
        # Set the network to evaluation mode (disables dropout, etc.)
        agent.network.eval()
        print(f"Successfully loaded DRL model from {model_path}")
    except Exception as e:
        print(f"Error loading DRL model: {e}. Skipping DRL agent run.")
        env.close()
        return

    episode_rewards = []
    episode_steps = []

    for episode in range(NUM_EPISODES):
        print(f"\n--- DRL: Starting Episode {episode + 1}/{NUM_EPISODES} ---")
        start_time = time.time()
        # Reset returns the *normalized* observation if normalize_obs=True
        observation, info = env.reset()
        current_episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        episode_log = [] # Log for the current episode

        while not terminated and not truncated:
            # DRL agent selects action based on potentially normalized observation
            action, _, _ = agent.get_action(observation)

            # Get the raw (unnormalized) observation from the info dict for logging purposes
            # This ensures logs contain human-readable values.
            current_raw_obs = info.get("raw_observation", observation)

            # Step the environment with the chosen action
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            current_episode_reward += reward
            step += 1

            # Log detailed step data if enabled (using raw observation)
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

            # Prepare for next iteration
            observation = next_observation # Use the potentially normalized observation for the next agent decision
            info = next_info # Update info dict

            # --- Check for episode end conditions --- #
            if terminated:
                print(f"DRL Episode finished after {step} steps (Terminated)")
            elif truncated:
                print(f"DRL Episode finished after {step} steps (Truncated - Max steps reached)")

        # --- End of Episode --- #
        end_time = time.time()
        print(f"DRL Episode {episode + 1} finished. Reward: {current_episode_reward:.2f}")
        # Retrieve final status from the last info dict
        print(f"  Final Status: {info.get('subsystem_statuses', 'N/A')}")
        print(f"  Duration: {end_time - start_time:.2f} seconds")

        episode_rewards.append(current_episode_reward)
        episode_steps.append(step)

        # Save detailed log for this episode
        if SAVE_DETAILED_LOGS:
            log_filename = os.path.join(LOGS_DIR, f"drl_episode_{episode}.json")
            try:
                with open(log_filename, 'w') as f:
                    json.dump(episode_log, f, indent=2)
            except Exception as e:
                print(f"Error saving detailed log {log_filename}: {e}")

    env.close()

    # Store aggregated results for this agent
    results_data['drl'] = {
        'rewards': episode_rewards,
        'steps': episode_steps
    }

    print(f"\n--- Simulation Summary (DRL Agent - {model_path}) ---")
    print(f"Episodes: {NUM_EPISODES}")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Avg Steps: {np.mean(episode_steps):.2f} ± {np.std(episode_steps):.2f}")


if __name__ == "__main__":
    # Dictionary to hold results from both agent runs
    results = {}

    # Run simulations for both agents
    run_classical_agent(results)
    run_drl_agent(results)

    # Save the combined summary results to a JSON file
    if results:
        try:
            # Ensure results directory exists
            os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
            with open(RESULTS_FILE, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nSummary results saved to {RESULTS_FILE}")
        except Exception as e:
            print(f"\nError saving summary results to {RESULTS_FILE}: {e}")
    else:
        print("\nNo summary results were generated to save.")

    if SAVE_DETAILED_LOGS:
        print(f"\nDetailed episode logs saved in directory: {LOGS_DIR}") 
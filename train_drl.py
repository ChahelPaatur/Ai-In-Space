import os
import numpy as np
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
import pandas as pd # For rolling average

# Use absolute imports assuming 'src' is in the Python path or run from project root
from src.spacecraft_env import SpacecraftEnv
from src.drl_agent import PPOAgent

# --- Training Parameters ---
TOTAL_TIMESTEPS = 50000  # Adjust as needed (e.g., 100k, 500k, 1M for better results)
STEPS_PER_UPDATE = 2048 # How often to run the PPO update (should match agent's buffer size ideally)
LEARNING_RATE = 3e-4
GAMMA = 0.99
PPO_EPSILON = 0.2
PPO_EPOCHS = 10
BATCH_SIZE = 64         # Used within agent.learn()
HIDDEN_SIZE = 64
SAVE_PATH = "ppo_agent.pth"
PRINT_INTERVAL = 10 # Print stats every N episodes (Increased interval)
LOG_INTERVAL_STEPS = STEPS_PER_UPDATE # Log average reward roughly every update cycle
PLOT_SAVE_DIR = "static/plots" # Directory to save the learning curve plot
LEARNING_CURVE_FILENAME = "learning_curve.png"
ROLLING_AVG_WINDOW = 100 # Window size for smoothing learning curve

def plot_learning_curve(timesteps, rewards, save_path, window=ROLLING_AVG_WINDOW):
    """Plots the learning curve with raw rewards and rolling average."""
    print(f"\nGenerating learning curve plot and saving to {save_path}...")
    if not timesteps or not rewards:
        print("No data to plot learning curve.")
        return

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 6))

    # Raw rewards (optional, can be noisy)
    # ax.plot(timesteps, rewards, label='Episode Reward', alpha=0.3, color='lightblue')

    # Rolling average
    if len(rewards) >= window:
        rewards_series = pd.Series(rewards)
        rolling_avg = rewards_series.rolling(window=window, min_periods=1).mean()
        ax.plot(timesteps, rolling_avg, label=f'Rolling Average Reward (Window={window})', color='cyan', linewidth=2)
    else:
         ax.plot(timesteps, rewards, label='Episode Reward (Not enough data for rolling avg)', alpha=0.8, color='cyan', linewidth=1.5)


    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("DRL Agent Learning Curve")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ensure plot directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        fig.savefig(save_path, dpi=300)
        print(f"Learning curve saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving learning curve plot: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory

def train():
    """Main training loop."""
    print("--- Starting DRL Agent Training ---")
    start_time = time.time()

    # --- Environment Setup ---
    # Ensure normalization is enabled as the agent likely expects it
    env = SpacecraftEnv(normalize_obs=True, max_steps=200) # Use max_steps consistent with comparison run
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Observation space size: {obs_size}")
    print(f"Action space size: {action_size}")

    # --- Agent Setup ---
    agent = PPOAgent(
        obs_size=obs_size,
        action_size=action_size,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        ppo_epsilon=PPO_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        device='cpu' # Or 'cuda' if available and desired
    )
    # Check if a pre-existing model exists to potentially resume training
    # agent.load_model(SAVE_PATH) # Uncomment to resume training

    # --- Training Loop ---
    observation, info = env.reset()
    current_total_steps = 0
    # Store history for learning curve plotting
    all_episode_rewards = []
    all_episode_timesteps = []
    # Store rewards of last N episodes for printing average
    recent_episode_rewards = deque(maxlen=ROLLING_AVG_WINDOW)
    current_episode_reward = 0
    current_episode_steps = 0
    completed_episodes = 0
    last_log_step = 0

    while current_total_steps < TOTAL_TIMESTEPS:
        # Get action from agent
        action, log_prob, value = agent.get_action(observation)

        # Step environment
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        current_episode_reward += reward
        current_episode_steps += 1
        current_total_steps += 1

        # Store experience
        done = terminated or truncated
        agent.store_experience(observation, action, reward, next_observation, done, log_prob, value)

        observation = next_observation
        info = next_info

        # --- Episode End Handling ---
        if done:
            completed_episodes += 1
            # Log data for plotting
            all_episode_rewards.append(current_episode_reward)
            all_episode_timesteps.append(current_total_steps)
            # Log data for printing recent average
            recent_episode_rewards.append(current_episode_reward)
            avg_reward = np.mean(recent_episode_rewards) if recent_episode_rewards else 0.0

            if completed_episodes % PRINT_INTERVAL == 0:
                print(f"Episode: {completed_episodes} | Steps: {current_total_steps}/{TOTAL_TIMESTEPS} | Ep Reward: {current_episode_reward:.2f} | Avg Reward (Last {ROLLING_AVG_WINDOW}): {avg_reward:.2f}")

            # Reset environment
            observation, info = env.reset()
            current_episode_reward = 0
            current_episode_steps = 0

        # --- PPO Update Handling ---
        # Check if enough steps collected since last update OR if buffer is full
        # The custom agent's learn method clears the buffer, so we call it when buffer full.
        if len(agent.memory) >= STEPS_PER_UPDATE:
            print(f"\nUpdating agent at step {current_total_steps} (Buffer size: {len(agent.memory)})...")
            agent.learn()
            print("Update complete.")
            # Log average reward around the time of update
            if recent_episode_rewards:
                 avg_reward_update = np.mean(recent_episode_rewards)
                 print(f"Average reward at update: {avg_reward_update:.2f}")
            last_log_step = current_total_steps


    # --- End of Training ---
    end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total Steps: {current_total_steps}")
    print(f"Total Episodes: {completed_episodes}")
    print(f"Training Time: {(end_time - start_time):.2f} seconds")

    # Save the final model
    agent.save_model(SAVE_PATH)

    # Plot learning curve
    plot_save_path = os.path.join(PLOT_SAVE_DIR, LEARNING_CURVE_FILENAME)
    plot_learning_curve(all_episode_timesteps, all_episode_rewards, plot_save_path)

    env.close()

if __name__ == "__main__":
    train() 
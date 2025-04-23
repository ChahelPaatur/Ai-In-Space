import os
import numpy as np
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
import pandas as pd # Required for efficient rolling average calculation

# Use absolute imports assuming 'src' is in the Python path or run from project root
from src.spacecraft_env import SpacecraftEnv
from src.drl_agent import PPOAgent

# --- Training Configuration ---
TOTAL_TIMESTEPS = 50000  # Total environment steps to train for
STEPS_PER_UPDATE = 2048 # Number of steps to collect before each PPO learning phase
LEARNING_RATE = 3e-4    # Learning rate for the Adam optimizer
GAMMA = 0.99            # Discount factor for future rewards
PPO_EPSILON = 0.2       # PPO clipping parameter
PPO_EPOCHS = 10         # Number of optimization epochs per PPO update
BATCH_SIZE = 64         # Minibatch size used within PPO epochs
HIDDEN_SIZE = 64        # Number of units in the MLP hidden layers
SAVE_PATH = "ppo_agent.pth" # File path to save the trained agent model
PRINT_INTERVAL = 10     # Frequency (in episodes) to print training progress
PLOT_SAVE_DIR = "static/plots" # Directory for saving the learning curve plot
LEARNING_CURVE_FILENAME = "learning_curve.png"
ROLLING_AVG_WINDOW = 100 # Window size for smoothing the learning curve plot

def plot_learning_curve(timesteps, rewards, save_path, window=ROLLING_AVG_WINDOW):
    """Generates and saves the learning curve plot (reward vs. timesteps)."""
    print(f"\nGenerating learning curve plot and saving to {save_path}...")
    if not timesteps or not rewards:
        print("No data available to plot learning curve.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate and plot rolling average for smoothed trend visualization
    if len(rewards) >= window:
        # Use pandas for efficient rolling calculation
        rewards_series = pd.Series(rewards)
        rolling_avg = rewards_series.rolling(window=window, min_periods=1).mean()
        ax.plot(timesteps, rolling_avg, label=f'Rolling Average Reward (Window={window})', color='cyan', linewidth=2)
    else:
        # Plot raw rewards if not enough data for meaningful rolling average
         ax.plot(timesteps, rewards, label='Episode Reward (Raw)', alpha=0.8, color='cyan', linewidth=1.5)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("DRL Agent Learning Curve")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Ensure the target directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        fig.savefig(save_path, dpi=300)
        print(f"Learning curve saved successfully to {save_path}")
    except Exception as e:
        print(f"Error saving learning curve plot: {e}")
    finally:
        plt.close(fig) # Prevent matplotlib memory leaks

def train():
    """Runs the main DRL agent training loop."""
    print("--- Starting DRL Agent Training ---")
    start_time = time.time()

    # Initialize the environment
    # Normalization should generally be enabled if the agent expects it
    env = SpacecraftEnv(normalize_obs=True, max_steps=200)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Observation space size: {obs_size}")
    print(f"Action space size: {action_size}")

    # Initialize the PPO agent
    agent = PPOAgent(
        obs_size=obs_size,
        action_size=action_size,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        ppo_epsilon=PPO_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        device='cpu' # Set device ('cpu' or 'cuda')
    )
    # Optional: Load existing model to resume training
    # if os.path.exists(SAVE_PATH):
    #     print(f"Loading existing model from {SAVE_PATH}")
    #     agent.load_model(SAVE_PATH)

    # --- Training Loop Initialization ---
    observation, info = env.reset()
    current_total_steps = 0
    # Logging lists for learning curve data
    all_episode_rewards = []
    all_episode_timesteps = []
    # Queue for calculating rolling average reward for print statements
    recent_episode_rewards = deque(maxlen=ROLLING_AVG_WINDOW)
    current_episode_reward = 0
    completed_episodes = 0

    print(f"Starting training loop for {TOTAL_TIMESTEPS} timesteps...")
    while current_total_steps < TOTAL_TIMESTEPS:
        # Agent selects action based on current observation
        action, log_prob, value = agent.get_action(observation)

        # Environment processes action and returns outcome
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        current_episode_reward += reward
        current_total_steps += 1

        # Store the transition in the agent's experience buffer
        done = terminated or truncated
        agent.store_experience(observation, action, reward, next_observation, done, log_prob, value)

        # Prepare for next iteration
        observation = next_observation

        # --- Episode Completion Handling ---
        if done:
            completed_episodes += 1
            all_episode_rewards.append(current_episode_reward)
            all_episode_timesteps.append(current_total_steps)
            recent_episode_rewards.append(current_episode_reward)

            # Print progress periodically
            if completed_episodes % PRINT_INTERVAL == 0:
                avg_reward = np.mean(recent_episode_rewards) if recent_episode_rewards else 0.0
                print(f"Episode: {completed_episodes} | Steps: {current_total_steps}/{TOTAL_TIMESTEPS} | Ep Reward: {current_episode_reward:.2f} | Avg Reward (Last {len(recent_episode_rewards)}): {avg_reward:.2f}")

            # Reset environment for the next episode
            observation, info = env.reset()
            current_episode_reward = 0

        # --- PPO Learning Update Trigger ---
        # Update the agent's policy and value networks when enough experience is collected
        if len(agent.memory) >= STEPS_PER_UPDATE:
            print(f"\nUpdating agent policy at step {current_total_steps}...")
            agent.learn() # This performs multiple epochs of optimization
            print("Agent update complete.")
            # Optional: Print average reward around the time of update
            # if recent_episode_rewards:
            #     avg_reward_update = np.mean(recent_episode_rewards)
            #     print(f"Average reward at update point: {avg_reward_update:.2f}")

    # --- End of Training Procedures ---
    end_time = time.time()
    print("\n--- Training Finished ---")
    print(f"Total Steps: {current_total_steps}")
    print(f"Total Episodes Completed: {completed_episodes}")
    print(f"Training Time: {(end_time - start_time):.2f} seconds")

    # Save the final trained model weights
    agent.save_model(SAVE_PATH)

    # Generate and save the learning curve plot
    plot_save_path = os.path.join(PLOT_SAVE_DIR, LEARNING_CURVE_FILENAME)
    plot_learning_curve(all_episode_timesteps, all_episode_rewards, plot_save_path)

    env.close() # Clean up environment resources

if __name__ == "__main__":
    train() 
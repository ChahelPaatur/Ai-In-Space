import os
import numpy as np
import torch
from collections import deque
import time
import matplotlib.pyplot as plt
import pandas as pd

# Use absolute imports assuming 'src' is in the Python path or run from project root
from src.spacecraft_env import SpacecraftEnv
from src.drl_agent import PPOAgent

# --- Training Configuration ---
TOTAL_TIMESTEPS = 1000000   # 1 million steps (20x the original)
STEPS_PER_UPDATE = 2048     # Number of steps to collect before each PPO learning phase
LEARNING_RATE = 3e-4        # Learning rate for the Adam optimizer
GAMMA = 0.99                # Discount factor for future rewards
PPO_EPSILON = 0.2           # PPO clipping parameter
PPO_EPOCHS = 10             # Number of optimization epochs per PPO update
BATCH_SIZE = 64             # Minibatch size used within PPO epochs
HIDDEN_SIZE = 128           # Increased number of units in the MLP hidden layers
SAVE_PATH = "ppo_agent_long.pth"  # New file path to save the trained agent model
INTERMEDIATE_SAVE_INTERVAL = 100000  # Save intermediate models every N steps
PRINT_INTERVAL = 10         # Frequency (in episodes) to print training progress
PLOT_SAVE_DIR = "static/plots"  # Directory for saving the learning curve plot
LEARNING_CURVE_FILENAME = "learning_curve_long.png"
ROLLING_AVG_WINDOW = 100    # Window size for smoothing the learning curve plot
CHECKPOINT_DIR = "models/checkpoints"  # Directory for saving intermediate models

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
    
    # Also plot raw rewards with transparency
    ax.scatter(timesteps, rewards, label='Episode Reward (Raw)', alpha=0.3, color='blue', s=10)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("DRL Agent Learning Curve (Long Training)")
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
        plt.close(fig)  # Prevent matplotlib memory leaks

def save_training_data(timesteps, rewards, filename="training_data.csv"):
    """Save training data to CSV for later analysis."""
    try:
        df = pd.DataFrame({
            'timestep': timesteps,
            'reward': rewards
        })
        os.makedirs(os.path.dirname(os.path.join(PLOT_SAVE_DIR, filename)), exist_ok=True)
        df.to_csv(os.path.join(PLOT_SAVE_DIR, filename), index=False)
        print(f"Saved training data to {os.path.join(PLOT_SAVE_DIR, filename)}")
    except Exception as e:
        print(f"Error saving training data: {e}")

def train_long():
    """Runs the main DRL agent training loop for extended training."""
    print("--- Starting Extended DRL Agent Training ---")
    start_time = time.time()

    # Initialize the environment
    env = SpacecraftEnv(normalize_obs=True, max_steps=200)
    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"Observation space size: {obs_size}")
    print(f"Action space size: {action_size}")

    # Initialize the PPO agent with larger network
    agent = PPOAgent(
        obs_size=obs_size,
        action_size=action_size,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        ppo_epsilon=PPO_EPSILON,
        ppo_epochs=PPO_EPOCHS,
        batch_size=BATCH_SIZE,
        hidden_size=HIDDEN_SIZE,
        device='cpu'  # Set device ('cpu' or 'cuda')
    )

    # Optional: Load existing model to resume training
    if os.path.exists("ppo_agent.pth"):
        print(f"Loading existing model from ppo_agent.pth to continue training")
        agent.load_model("ppo_agent.pth")

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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
            agent.learn()  # This performs multiple epochs of optimization
            print("Agent update complete.")

        # --- Save intermediate models ---
        if current_total_steps % INTERMEDIATE_SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ppo_agent_step_{current_total_steps}.pth")
            agent.save_model(checkpoint_path)
            print(f"Saved intermediate model at step {current_total_steps} to {checkpoint_path}")
            
            # Also update the learning curve periodically
            plot_save_path = os.path.join(PLOT_SAVE_DIR, LEARNING_CURVE_FILENAME)
            plot_learning_curve(all_episode_timesteps, all_episode_rewards, plot_save_path)
            save_training_data(all_episode_timesteps, all_episode_rewards)

    # --- End of Training Procedures ---
    end_time = time.time()
    training_duration = end_time - start_time
    print("\n--- Training Finished ---")
    print(f"Total Steps: {current_total_steps}")
    print(f"Total Episodes Completed: {completed_episodes}")
    print(f"Training Time: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")

    # Save the final trained model weights
    agent.save_model(SAVE_PATH)
    print(f"Saved final model to {SAVE_PATH}")

    # Generate and save the learning curve plot
    plot_save_path = os.path.join(PLOT_SAVE_DIR, LEARNING_CURVE_FILENAME)
    plot_learning_curve(all_episode_timesteps, all_episode_rewards, plot_save_path)
    
    # Save training data for further analysis
    save_training_data(all_episode_timesteps, all_episode_rewards)

    env.close()  # Clean up environment resources

if __name__ == "__main__":
    train_long() 
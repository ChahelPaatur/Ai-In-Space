import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import os
from collections import deque
import random

class ActorCriticNetwork(nn.Module):
    """Simple MLP Actor-Critic Network."""
    def __init__(self, obs_size, action_size, hidden_size=64):
        super().__init__()

        # Shared layers (optional)
        self.shared_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor head (outputs action probabilities)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1) # Probabilities for discrete actions
        )

        # Critic head (outputs state value)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        shared_features = self.shared_net(x)
        action_probs = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_probs, state_value

class PPOAgent:
    """PPO Agent Implementation."""
    def __init__(self, obs_size, action_size, lr=3e-4, gamma=0.99, ppo_epsilon=0.2, ppo_epochs=10, batch_size=64, hidden_size=64, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.gamma = gamma
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.network = ActorCriticNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Simple experience buffer (replace with more sophisticated one if needed)
        self.memory = deque(maxlen=2048) # Example buffer size

    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in the buffer."""
        # Ensure tensors are on the correct device and detached
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).detach()
        action = torch.as_tensor([action], dtype=torch.int64, device=self.device).detach()
        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device).detach()
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).detach()
        done = torch.as_tensor([done], dtype=torch.float32, device=self.device).detach()
        log_prob = torch.as_tensor([log_prob], dtype=torch.float32, device=self.device).detach()
        value = torch.as_tensor([value], dtype=torch.float32, device=self.device).detach()

        self.memory.append((state, action, reward, next_state, done, log_prob, value))

    def get_action(self, observation: np.ndarray) -> tuple[int, float, float]:
        """Select an action based on the current policy (actor network)."""
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.network.eval() # Set network to evaluation mode
        with torch.no_grad():
            action_probs, state_value = self.network(state)
        self.network.train() # Set network back to training mode

        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), state_value.item()

    def calculate_advantages(self, rewards, values, dones, last_value):
        """Calculate advantages using simple TD(0) approach."""
        # Note: GAE (Generalized Advantage Estimation) is often preferred
        advantages = torch.zeros_like(rewards)
        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t+1] # Use done flag of *next* state
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = delta # Simple TD error as advantage
            # If using GAE: advantages[t] = delta + self.gamma * self.lambda_gae * next_non_terminal * last_adv
            # last_adv = advantages[t]
        returns = advantages + values # Q-value estimate Q(s,a) approx= A(s,a) + V(s)
        return advantages, returns

    def learn(self):
        """Perform the PPO update step using collected experience."""
        if len(self.memory) < self.batch_size:
            return # Not enough samples yet

        # Sample a batch from memory
        # For simplicity, use the whole buffer if it's large enough, or sample
        # A true implementation often uses all data from the last rollout (e.g., 2048 steps)
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Convert batch to tensors
        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        # next_states = torch.stack(next_states).to(self.device) # Not directly needed for TD(0) advantage calc this way
        dones = torch.cat(dones).to(self.device)
        old_log_probs = torch.cat(old_log_probs).to(self.device)
        old_values = torch.cat(old_values).to(self.device).squeeze() # Ensure correct shape

        # Calculate advantages and returns
        # Need value of the last state in the batch for proper calculation if episode didn't end
        # Simplified: Use old_values directly. More correctly, re-calculate values or use GAE.
        advantages, returns = self.calculate_advantages(rewards, old_values, dones, 0.0) # Assuming last_value=0 if done

        # Normalize advantages (optional but recommended)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update Loop
        for _ in range(self.ppo_epochs):
            # Get current policy probabilities and values
            action_probs, current_values = self.network(states)
            current_values = current_values.squeeze()
            dist = Categorical(probs=action_probs)
            current_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean() # For entropy bonus

            # Calculate PPO ratio
            ratio = torch.exp(current_log_probs - old_log_probs)

            # Calculate surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss (MSE)
            value_loss = nn.functional.mse_loss(current_values, returns)

            # Total loss
            # Entropy bonus encourages exploration (coefficient needs tuning)
            entropy_coeff = 0.01
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient clipping
            # nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Clear memory after update (or use a more sophisticated buffer management)
        self.memory.clear()

    def save_model(self, path="ppo_agent.pth"):
        """Save the network weights."""
        print(f"Saving model to {path}")
        torch.save(self.network.state_dict(), path)

    def load_model(self, path="ppo_agent.pth"):
        """Load the network weights."""
        if os.path.exists(path):
            print(f"Loading model from {path}")
            self.network.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Warning: Model file not found at {path}") 
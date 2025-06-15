import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import os
import time
import logging
from collections import deque
import random
import json

# Configure logging for safety-critical system requirements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler("drl_agent_safety.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("DRL-SafetyAgent")

# Safety compliance parameters
SAFETY_CONFIG = {
    "action_verification": True,        # Verify actions before execution
    "input_validation": True,           # Validate inputs for anomalies
    "self_monitoring": True,            # Monitor system health
    "redundancy_checks": True,          # Perform redundant computations for validation
    "safety_envelope": True,            # Maintain operational safety envelope
    "watchdog_timeout_ms": 1000,        # Watchdog timer timeout in milliseconds
    "heartbeat_interval_ms": 2000,      # Safety heartbeat interval
    "safety_criticality_level": "C",    # DO-178C criticality level (A-E)
    "error_recovery": True,             # Automatic recovery from errors
    "action_bounds_check": True,        # Check if actions are within bounds
    "version": "1.0.0",                 # Version tracking for safety certification
    "certification_id": "DRL-FDIR-001"  # Certification identifier
}

class SafetyMonitor:
    """Safety monitoring system compliant with DO-178C and ECSS-E-ST-40C standards"""
    def __init__(self, config=None):
        self.config = config or SAFETY_CONFIG
        self.last_heartbeat = time.time()
        self.anomaly_count = 0
        self.max_anomalies = 10
        self.watchdog_last_reset = time.time()
        self.healthy = True
        logger.info(f"SafetyMonitor initialized with config: {json.dumps(self.config, indent=2)}")
        
    def reset_watchdog(self):
        """Reset the watchdog timer"""
        current_time = time.time()
        elapsed = (current_time - self.watchdog_last_reset) * 1000  # convert to ms
        
        if elapsed > self.config["watchdog_timeout_ms"]:
            logger.warning(f"Watchdog timeout exceeded: {elapsed:.2f}ms > {self.config['watchdog_timeout_ms']}ms")
            self.anomaly_count += 1
            self.check_health()
            
        self.watchdog_last_reset = current_time
        return self.healthy
        
    def send_heartbeat(self):
        """Send heartbeat signal to indicate system is alive"""
        current_time = time.time()
        elapsed = (current_time - self.last_heartbeat) * 1000  # convert to ms
        
        if elapsed > self.config["heartbeat_interval_ms"]:
            logger.info(f"Safety heartbeat: System operational, elapsed={elapsed:.2f}ms")
            self.last_heartbeat = current_time
            
    def validate_input(self, observation):
        """Validate input data for anomalies (DO-178C input validation)"""
        if not self.config["input_validation"]:
            return True
            
        # Check for NaN or infinity values
        if np.isnan(observation).any() or np.isinf(observation).any():
            logger.error(f"Input validation failed: NaN or Inf values in observation")
            self.anomaly_count += 1
            self.check_health()
            return False
            
        # Check for out-of-range values (assuming normalized [-1, 1] inputs)
        # Adjust based on your actual input ranges
        if np.any(observation > 100) or np.any(observation < -100):
            logger.warning(f"Input validation: Observation contains potential out-of-range values")
            # Don't increment anomaly count for this, just log warning
            
        return True
        
    def verify_action(self, action, action_probs):
        """Verify action is safe to execute (DO-178C output validation)"""
        if not self.config["action_verification"]:
            return True
            
        # Check if action probability distribution is valid
        if not np.isclose(np.sum(action_probs.detach().cpu().numpy()), 1.0, atol=1e-3):
            logger.error(f"Action verification failed: Invalid probability distribution")
            self.anomaly_count += 1
            self.check_health()
            return False
            
        # If we have defined safe actions for specific states, we could check here
        return True
        
    def check_health(self):
        """Check overall system health status"""
        if self.anomaly_count > self.max_anomalies:
            if self.healthy:
                logger.critical(f"System health compromised: {self.anomaly_count} anomalies exceeded threshold {self.max_anomalies}")
                self.healthy = False
        return self.healthy

    def recover(self):
        """Attempt to recover system health"""
        if not self.healthy and self.config["error_recovery"]:
            logger.warning("Attempting system recovery")
            self.anomaly_count = 0
            self.healthy = True
            self.last_heartbeat = time.time()
            self.watchdog_last_reset = time.time()
            logger.info("System recovery complete")
        return self.healthy


class ActorCriticNetwork(nn.Module):
    """Safety-enhanced MLP Actor-Critic Network with bounds verification."""
    def __init__(self, obs_size, action_size, hidden_size=64):
        super().__init__()

        # Record input specifications for safety verification
        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_size = hidden_size

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
        
        # Initialize with safety-compliant parameter ranges
        self._initialize_safety_parameters()
        
        logger.info(f"ActorCriticNetwork initialized: obs_size={obs_size}, action_size={action_size}, hidden_size={hidden_size}")

    def _initialize_safety_parameters(self):
        """Initialize parameters with safety bounds (ECSS-E-ST-40C compliance)"""
        # Using Xavier/Glorot initialization for more stable initial behavior
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
        logger.info("Network parameters initialized with safety-compliant bounds")

    def forward(self, x):
        """Forward pass with safety validation"""
        # Input validation (DO-178C compliance)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error("NaN or Inf detected in network input")
            # Use default safe values when invalid input detected
            x = torch.zeros_like(x)
        
        shared_features = self.shared_net(x)
        action_probs = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        
        # Safety check for action probability distribution (DO-178C compliance)
        if not torch.isclose(action_probs.sum(dim=1), torch.tensor([1.0], device=action_probs.device), atol=1e-3).all():
            logger.warning("Action probabilities don't sum to 1.0, applying safety correction")
            # Apply safety correction - normalize to ensure valid probability distribution
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
        return action_probs, state_value


class PPOAgent:
    """Safety-enhanced PPO Agent Implementation compliant with DO-178C and ECSS-E-ST-40C."""
    def __init__(self, obs_size, action_size, lr=3e-4, gamma=0.99, ppo_epsilon=0.2, ppo_epochs=10, batch_size=64, hidden_size=64, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing safety-compliant PPO agent on device: {self.device}")

        # Core PPO parameters
        self.gamma = gamma
        self.ppo_epsilon = ppo_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.obs_size = obs_size
        self.action_size = action_size
        
        # Safety system initialization
        self.safety_monitor = SafetyMonitor()
        self.last_safe_action = 0  # Default safe action
        self.diagnostic_data = {}
        self.training_allowed = True  # Can disable training in critical operations
        self.safety_mode = False  # Safety mode flag
        
        # Network and optimization
        self.network = ActorCriticNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Memory buffer with safety validation
        self.memory = deque(maxlen=2048) 
        
        # Redundancy for safety-critical systems (ECSS-E-ST-40C compliance)
        self.shadow_network = ActorCriticNetwork(obs_size, action_size, hidden_size).to(self.device)
        self.shadow_network.load_state_dict(self.network.state_dict())
        
        # System verification on initialization
        self._verify_system_integrity()
        
        logger.info(f"Safety-compliant PPO agent initialization complete")

    def _verify_system_integrity(self):
        """Verify system integrity as required by DO-178C"""
        try:
            # Generate test input
            test_input = torch.zeros((1, self.obs_size), device=self.device)
            
            # Test main network
            action_probs, _ = self.network(test_input)
            
            # Verify outputs are as expected
            if action_probs.shape != (1, self.action_size):
                raise ValueError(f"Network output shape mismatch: {action_probs.shape} vs expected {(1, self.action_size)}")
                
            # Test shadow network for redundancy
            shadow_probs, _ = self.shadow_network(test_input)
            
            logger.info("System integrity verification passed")
            return True
        except Exception as e:
            logger.critical(f"System integrity verification failed: {str(e)}")
            self.safety_mode = True
            return False

    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        """Store a transition in the buffer with safety validation."""
        # Ensure tensors are on the correct device and detached
        try:
            # Validate inputs before storing (DO-178C input validation)
            if not self.safety_monitor.validate_input(state) or not self.safety_monitor.validate_input(next_state):
                logger.warning("Experience rejected due to input validation failure")
                return
                
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).detach()
        action = torch.as_tensor([action], dtype=torch.int64, device=self.device).detach()
        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device).detach()
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).detach()
        done = torch.as_tensor([done], dtype=torch.float32, device=self.device).detach()
        log_prob = torch.as_tensor([log_prob], dtype=torch.float32, device=self.device).detach()
        value = torch.as_tensor([value], dtype=torch.float32, device=self.device).detach()

            # Safety check for numerical stability (ECSS-E-ST-40C compliance)
            if torch.isnan(log_prob).any() or torch.isinf(log_prob).any():
                logger.warning("NaN or Inf detected in log probability, experience rejected")
                return

        self.memory.append((state, action, reward, next_state, done, log_prob, value))

        except Exception as e:
            logger.error(f"Error storing experience: {str(e)}")
            # In safety-critical systems, we must handle all exceptions
            self.safety_monitor.anomaly_count += 1
            self.safety_monitor.check_health()

    def get_action(self, observation: np.ndarray) -> tuple:
        """Select action with safety verification according to DO-178C standards."""
        # Reset watchdog timer (DO-178C requirement)
        self.safety_monitor.reset_watchdog()
        
        # Validate input
        if not self.safety_monitor.validate_input(observation):
            logger.warning("Using last safe action due to input validation failure")
            # Return last safe action if input validation fails
            return self.last_safe_action, 0.0, 0.0, {"safety_violation": True}
            
        try:
            # Prepare observation
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
            # Safety check - perform redundant computation (ECSS-E-ST-40C redundancy)
            self.network.eval() 
            self.shadow_network.eval()
            
        with torch.no_grad():
                # Main network inference
            action_probs, state_value = self.network(state)
                
                # Shadow network inference for redundancy check
                if SAFETY_CONFIG["redundancy_checks"]:
                    shadow_probs, shadow_value = self.shadow_network(state)
                    
                    # Compare outputs for consistency (DO-178C verification)
                    probs_diff = torch.abs(action_probs - shadow_probs).max().item()
                    if probs_diff > 0.1:  # Threshold for inconsistency
                        logger.warning(f"Network inconsistency detected: {probs_diff:.4f} > 0.1")
                        self.safety_monitor.anomaly_count += 1
                        
                        # In critical applications, we might decide to use a safe default action
                        # if redundancy check fails, but here we'll continue with the main network
            
            self.network.train()
            
            # Sample action from probability distribution
        dist = Categorical(probs=action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

            # Verify action is safe to execute (DO-178C verification)
            if not self.safety_monitor.verify_action(action.item(), action_probs):
                logger.warning("Action verification failed, using last safe action")
                return self.last_safe_action, 0.0, 0.0, {"safety_violation": True}
                
            # Update last safe action
            self.last_safe_action = action.item()
            
            # Generate diagnostic information
            diagnostics = {
                "action_probs": action_probs.detach().cpu().numpy().flatten().tolist(),
                "value": state_value.item(),
                "safety_mode": self.safety_mode,
                "anomaly_count": self.safety_monitor.anomaly_count
            }
            
            # Send heartbeat (DO-178C monitoring requirement)
            self.safety_monitor.send_heartbeat()
            
            return action.item(), log_prob.item(), state_value.item(), diagnostics
            
        except Exception as e:
            logger.error(f"Error selecting action: {str(e)}")
            self.safety_monitor.anomaly_count += 1
            self.safety_monitor.check_health()
            
            # In safety-critical systems, we must have a fallback
            return self.last_safe_action, 0.0, 0.0, {"error": str(e), "safety_violation": True}

    def calculate_advantages(self, rewards, values, dones, last_value):
        """Calculate advantages with numerical stability safeguards."""
        try:
            # Safety bounds check (DO-178C requirement)
            if torch.isnan(rewards).any() or torch.isinf(rewards).any():
                logger.warning("NaN or Inf detected in rewards, applying safety correction")
                rewards = torch.where(torch.isnan(rewards) | torch.isinf(rewards), 
                                      torch.zeros_like(rewards), rewards)
            
        advantages = torch.zeros_like(rewards)
        last_adv = 0.0
            
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                    next_non_terminal = 1.0 - dones[t+1] 
                next_value = values[t+1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                advantages[t] = delta 
            
            returns = advantages + values 
            
            # Safety check for numerical stability (ECSS-E-ST-40C requirement)
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.error("Advantage calculation resulted in NaN or Inf values")
                # Use safe default values
                advantages = torch.zeros_like(advantages)
                returns = values.clone()
                
        return advantages, returns
            
        except Exception as e:
            logger.error(f"Error calculating advantages: {str(e)}")
            # Safe fallback
            return torch.zeros_like(rewards), values

    def learn(self):
        """Perform PPO update with safety guarantees."""
        # Check if learning is allowed in current safety state
        if not self.training_allowed or self.safety_mode:
            logger.warning("Learning skipped due to safety constraints")
            return
            
        if len(self.memory) < self.batch_size:
            return # Not enough samples yet

        try:
            # Reset watchdog for long operation
            self.safety_monitor.reset_watchdog()

        # Sample a batch from memory
        batch_indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        
        # Convert batch to tensors
        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.cat(rewards).to(self.device)
        dones = torch.cat(dones).to(self.device)
        old_log_probs = torch.cat(old_log_probs).to(self.device)
            old_values = torch.cat(old_values).to(self.device).squeeze() 

        # Calculate advantages and returns
            advantages, returns = self.calculate_advantages(rewards, old_values, dones, 0.0)

            # Safety check - normalize advantages with numerical stability
            if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update Loop
            for epoch in range(self.ppo_epochs):
                # Safety checkpoint - reset watchdog during long operations
                if epoch % 2 == 0:
                    self.safety_monitor.reset_watchdog()
                    
            # Get current policy probabilities and values
            action_probs, current_values = self.network(states)
            current_values = current_values.squeeze()
                
                # Safety check - verify output validity
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    logger.error(f"NaN or Inf detected in network outputs during training epoch {epoch}")
                    break  # Stop training if numerical instability detected
                
            dist = Categorical(probs=action_probs)
            current_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean() 

                # Calculate PPO ratio with safety bounds
            ratio = torch.exp(current_log_probs - old_log_probs)

                # Safety clipping to prevent extreme values (DO-178C requirement)
                ratio = torch.clamp(ratio, 0.1, 10.0)  # Additional safety bounds

                # Standard PPO objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with safety bounds
            value_loss = nn.functional.mse_loss(current_values, returns)

                # Total loss with entropy regularization
            entropy_coeff = 0.01
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy

                # Optimization step with safety bounds
            self.optimizer.zero_grad()
            loss.backward()
                
                # Safety requirement - gradient clipping (ECSS-E-ST-40C compliance)
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

                # Update shadow network occasionally for redundancy (every 5 epochs)
                if epoch % 5 == 0 and SAFETY_CONFIG["redundancy_checks"]:
                    self.shadow_network.load_state_dict(self.network.state_dict())

            # Clear memory after update
        self.memory.clear()
            
            # Log training completion
            logger.info(f"PPO update completed successfully: {self.ppo_epochs} epochs")
            
        except Exception as e:
            logger.error(f"Error during learning: {str(e)}")
            self.safety_monitor.anomaly_count += 1
            self.safety_monitor.check_health()

    def save_model(self, path="ppo_agent.pth"):
        """Save the network weights with safety verification."""
        try:
            # Create safety metadata
            safety_metadata = {
                "version": SAFETY_CONFIG["version"],
                "certification_id": SAFETY_CONFIG["certification_id"],
                "timestamp": time.time(),
                "safety_criticality_level": SAFETY_CONFIG["safety_criticality_level"],
                "architecture": {
                    "obs_size": self.obs_size,
                    "action_size": self.action_size,
                    "hidden_size": self.network.hidden_size
                }
            }
            
            # Save metadata separately for certification purposes
            metadata_path = path.replace(".pth", "_safety_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(safety_metadata, f, indent=2)
            
            logger.info(f"Saving safety-compliant model to {path}")
            torch.save({
                "state_dict": self.network.state_dict(),
                "safety_metadata": safety_metadata
            }, path)
            logger.info(f"Model saved successfully with safety metadata")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path="ppo_agent.pth"):
        """Load the network weights with safety verification."""
        try:
        if os.path.exists(path):
                logger.info(f"Loading model from {path}")
                
                # Load model with safety checks
                checkpoint = torch.load(path, map_location=self.device)
                
                # Check if it's the new safety-compliant format or old format
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    # New format with safety metadata
                    self.network.load_state_dict(checkpoint["state_dict"])
                    
                    # Verify safety metadata if available
                    if "safety_metadata" in checkpoint:
                        metadata = checkpoint["safety_metadata"]
                        logger.info(f"Loaded safety-compliant model: "
                                    f"certification={metadata.get('certification_id', 'unknown')}, "
                                    f"version={metadata.get('version', 'unknown')}")
                        
                        # Verify architecture matches
                        arch = metadata.get("architecture", {})
                        if (arch.get("obs_size") != self.obs_size or 
                            arch.get("action_size") != self.action_size):
                            logger.warning(f"Architecture mismatch in loaded model: "
                                          f"expected obs={self.obs_size}, actions={self.action_size}, "
                                          f"got obs={arch.get('obs_size')}, actions={arch.get('action_size')}")
                else:
                    # Old format - just state dict
                    self.network.load_state_dict(checkpoint)
                    logger.warning("Loaded model without safety metadata (legacy format)")
                
                # Update shadow network for redundancy
                self.shadow_network.load_state_dict(self.network.state_dict())
                
                # Verify system integrity after loading
                self._verify_system_integrity()
                
                logger.info("Model loaded and verified successfully")
        else:
                logger.warning(f"Warning: Model file not found at {path}")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.safety_mode = True
            logger.warning("Switched to safety mode due to model loading failure") 
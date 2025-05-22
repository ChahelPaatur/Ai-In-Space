import numpy as np
import torch

from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent

class HybridFDIRAgent:
    """
    A hybrid FDIR agent that combines rule-based and DRL approaches.
    
    This agent uses a rule-based system as a safety baseline and allows
    the DRL agent to override decisions in certain conditions, creating
    a more robust fault management system.
    """
    
    def __init__(self, obs_size, action_size, rule_thresholds=None, drl_model_path="ppo_agent.pth", 
                 confidence_threshold=0.7, device='cpu'):
        """
        Initialize the hybrid agent with both rule-based and DRL components.
        
        Args:
            obs_size (int): Size of the observation space
            action_size (int): Size of the action space
            rule_thresholds (dict, optional): Custom thresholds for rule-based agent
            drl_model_path (str): Path to the trained DRL model
            confidence_threshold (float): Minimum confidence for DRL to override rules
            device (str): Device to run DRL model on ('cpu' or 'cuda')
        """
        # Initialize the rule-based component
        self.rule_agent = RuleBasedFDIR(thresholds=rule_thresholds)
        
        # Initialize the DRL component
        self.drl_agent = PPOAgent(obs_size, action_size, device=device)
        self.drl_agent.load_model(drl_model_path)
        self.drl_agent.network.eval()  # Set to evaluation mode
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.action_size = action_size
        self.device = device
        
        # Metrics tracking
        self.decision_source_history = []  # Track which system made decisions
        self.override_count = 0           # Count of DRL overrides
        self.rule_count = 0               # Count of rule-based decisions
        
        # Safety-critical actions that rule-based should handle
        self.safety_critical_actions = [
            self.rule_agent.ACTION_MAP["RECOVER_EPS"],
            self.rule_agent.ACTION_MAP["ENTER_SAFE_MODE"]
        ]
        
        # Mapping for decision sources (for logging)
        self.decision_source_map = {
            "rule": "Rule-Based",
            "drl": "DRL",
            "rule_override": "Rule (Safety Override)",
            "drl_confident": "DRL (High Confidence)"
        }
    
    def get_action(self, observation, info):
        """
        Determine action using both rule-based and DRL systems.
        
        Args:
            observation (np.ndarray): The current environment observation
            info (dict): Additional environment information
            
        Returns:
            int: The action to take
            dict: Additional information about decision process
        """
        # Get rule-based agent's decision
        rule_action = self.rule_agent.get_action(observation, info)
        
        # Get DRL agent's decision with probabilities
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.drl_agent.network(state)
        
        # Convert to numpy for easier handling
        action_probs = action_probs.squeeze().cpu().numpy()
        drl_action = np.argmax(action_probs)
        drl_confidence = action_probs[drl_action]
        
        # Decision logic
        decision_source = "rule"  # Default
        final_action = rule_action
        
        # Case 1: If rule-based system chooses a safety-critical action, always use it
        if rule_action in self.safety_critical_actions:
            decision_source = "rule_override"
            final_action = rule_action
        
        # Case 2: If DRL is confident and not contradicting a safety action, use DRL
        elif drl_confidence >= self.confidence_threshold:
            decision_source = "drl_confident"
            final_action = drl_action
        
        # Case 3: Rule-based subsystem/mode decision vs DRL action control decision
        # This is where more sophisticated logic could be implemented
        # For example, using rule-based for system mode changes but DRL for actuator control
        elif rule_action == self.rule_agent.ACTION_MAP["NO_OP"]:
            # If rule-based has no specific action, defer to DRL
            decision_source = "drl"
            final_action = drl_action
        
        # Update metrics
        self.decision_source_history.append(decision_source)
        if "drl" in decision_source:
            self.override_count += 1
        else:
            self.rule_count += 1
        
        # Additional information for logging/analysis
        decision_info = {
            "rule_action": rule_action,
            "drl_action": int(drl_action),
            "drl_confidence": float(drl_confidence),
            "decision_source": self.decision_source_map.get(decision_source, decision_source),
            "override_count": self.override_count,
            "rule_count": self.rule_count
        }
        
        return final_action, decision_info 
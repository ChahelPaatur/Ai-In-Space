import numpy as np
import torch

from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent

class HybridFDIRAgent:
    """
    A hybrid FDIR agent that combines rule-based and DRL approaches.
    
    # Paper reference: Section 3.4 "Hybrid Agent (HybridFDIRAgent)" - This class implements
    # the novel hybrid architecture described in the paper that combines the deterministic
    # safety guarantees of rule-based systems with the adaptability of DRL.
    
    This agent uses a rule-based system as a safety baseline and allows
    the DRL agent to override decisions in certain conditions, creating
    a more robust fault management system.
    """
    
    def __init__(self, obs_size, action_size, rule_thresholds=None, drl_model_path="ppo_agent.pth", 
                 confidence_threshold=0.7, device='cpu'):
        """
        Initialize the hybrid agent with both rule-based and DRL components.
        
        # Paper reference: Section 3.4 "Decision Architecture" - The confidence threshold
        # parameter implements the arbitration mechanism shown in Figure 8b, determining
        # when DRL decisions override rule-based recommendations.
        
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
        
        # Recovery cooldown mechanism to reduce false positives
        # Paper reference: Section 5.4 "False Positive Handling" - Post-study experiments
        # showed that implementing a cooldown period after recovery actions dramatically
        # reduced false positives during system stabilization
        self.recovery_cooldown = 0  # Steps to wait after a recovery action
        self.cooldown_period = 20   # Default cooldown period (20 steps)
        self.last_action = None     # Track the last action taken
        
        # Adaptive confidence threshold - increases temporarily after recovery actions
        # Paper reference: Section 5.4 - "A more sophisticated approach might incorporate
        # temporal context and subsystem-specific confidence thresholds"
        self.base_confidence_threshold = confidence_threshold
        self.current_confidence_threshold = confidence_threshold
        self.subsystem_status = {'EPS': 'Nominal', 'ADCS': 'Nominal', 'TCS': 'Nominal'}
        
        # Metrics tracking
        # Paper reference: Section 4.2 "Hybrid Decision Distribution" - These metrics
        # enable the analysis of decision sources shown in Figure 6 of the paper,
        # tracking which component (rule-based or DRL) makes each decision
        self.decision_source_history = []  # Track which system made decisions
        self.override_count = 0           # Count of DRL overrides
        self.rule_count = 0               # Count of rule-based decisions
        
        # Safety-critical actions that rule-based should handle
        # Paper reference: Section 3.4 "Decision Architecture" - Rule-based decisions
        # always override for safety-critical actions as described in the paper
        self.safety_critical_actions = [
            self.rule_agent.ACTION_MAP["RECOVER_EPS"],
            self.rule_agent.ACTION_MAP["RECOVER_ADCS"],
            self.rule_agent.ACTION_MAP["RECOVER_TCS"],
            self.rule_agent.ACTION_MAP["ENTER_SAFE_MODE"]
        ]
        
        # Recovery actions (used for cooldown)
        self.recovery_actions = [
            self.rule_agent.ACTION_MAP["RECOVER_EPS"],
            self.rule_agent.ACTION_MAP["RECOVER_ADCS"],
            self.rule_agent.ACTION_MAP["RECOVER_TCS"]
        ]
        
        # Mapping for decision sources (for logging)
        # Paper reference: Section 4.2 "Hybrid Decision Distribution" - The four decision
        # sources directly correspond to those shown in Figure 6 (Rule-based, DRL,
        # Rule-based Safety Override, and DRL High Confidence)
        self.decision_source_map = {
            "rule": "Rule-Based",
            "drl": "DRL",
            "rule_override": "Rule (Safety Override)",
            "drl_confident": "DRL (High Confidence)",
            "cooldown": "Rule (Cooldown Period)"
        }
    
    def get_action(self, observation, info):
        """
        Determine action using both rule-based and DRL systems.
        
        # Paper reference: Section 3.4 "Decision Architecture" - This function implements
        # the sophisticated arbitration mechanism illustrated in Figure 8b that determines
        # which component makes the final decision based on safety criticality, DRL
        # confidence, and rule-based defaults.
        
        Args:
            observation (np.ndarray): The current environment observation
            info (dict): Additional environment information
            
        Returns:
            int: The action to take
            dict: Additional information about decision process
        """
        # Update subsystem status from info
        if 'subsystem_statuses' in info:
            self.subsystem_status = info['subsystem_statuses']
        
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
        
        # Dynamic confidence threshold based on subsystem status
        # Use higher threshold if any subsystem is in fault condition
        # Paper reference: Section 5.4 - "A more sophisticated approach might incorporate 
        # system state-based confidence thresholds"
        self.current_confidence_threshold = self.base_confidence_threshold
        for subsys, status in self.subsystem_status.items():
            if status != 'Nominal':
                # Increase threshold when faults are present to reduce false positives
                self.current_confidence_threshold = min(0.95, self.base_confidence_threshold + 0.1)
                break
        
        # Handle recovery cooldown to prevent unnecessary repeated recoveries
        # Paper reference: Section 5.4 - "Introducing a short 'cooldown period' after
        # recovery actions could prevent the cascade of false positives we observed"
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
            decision_source = "cooldown"
            
            # During cooldown, only allow recovery actions from rule-based agent
            if rule_action in self.safety_critical_actions:
                final_action = rule_action
            else:
                # Otherwise, prefer conservative NO_OP during stabilization
                final_action = self.rule_agent.ACTION_MAP["NO_OP"]
        else:
            # Regular decision logic when not in cooldown
            # Paper reference: Section 3.4 "Decision Architecture" - Implements the three-part
            # decision logic described in the paper with additional confidence adaptations
            decision_source = "rule"  # Default
            final_action = rule_action
            
            # Case 1: If rule-based system chooses a safety-critical action, always use it
            if rule_action in self.safety_critical_actions:
                decision_source = "rule_override"
                final_action = rule_action
                
                # Start cooldown after recovery actions to reduce false positives
                if rule_action in self.recovery_actions:
                    self.recovery_cooldown = self.cooldown_period
            
            # Case 2: If DRL is confident and not contradicting a safety action, use DRL
            elif drl_confidence >= self.current_confidence_threshold:
                decision_source = "drl_confident"
                final_action = drl_action
                
                # Start cooldown if DRL suggests a recovery action
                if drl_action in self.recovery_actions:
                    self.recovery_cooldown = self.cooldown_period
            
            # Case 3: Use DRL for heater control and gyro bias reset
            # Paper reference: Section 4.2 - "The DRL agent utilized a much broader action
            # repertoire than the Classical agent, frequently employing actions like HeaterON/OFF"
            elif drl_action in [4, 5, 6]:  # Heater and gyro reset actions
                if drl_confidence >= (self.current_confidence_threshold - 0.1):  # Lower threshold for these actions
                    decision_source = "drl"
                    final_action = drl_action
            
            # Case 4: Rule-based default in nominal case
            elif rule_action == self.rule_agent.ACTION_MAP["NO_OP"]:
                # If rule-based has no specific action, defer to DRL if it has strong preference
                if drl_confidence >= (self.current_confidence_threshold - 0.05):
                    decision_source = "drl"
                    final_action = drl_action
        
        # Update metrics
        self.decision_source_history.append(decision_source)
        if "drl" in decision_source:
            self.override_count += 1
        else:
            self.rule_count += 1
        
        # Store last action for context in future decisions
        self.last_action = final_action
        
        # Additional information for logging/analysis
        decision_info = {
            "rule_action": rule_action,
            "drl_action": int(drl_action),
            "drl_confidence": float(drl_confidence),
            "confidence_threshold": float(self.current_confidence_threshold),
            "cooldown_remaining": self.recovery_cooldown,
            "decision_source": self.decision_source_map.get(decision_source, decision_source),
            "override_count": self.override_count,
            "rule_count": self.rule_count,
            "subsystem_status": self.subsystem_status
        }
        
        return final_action, decision_info 
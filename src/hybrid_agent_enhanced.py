import numpy as np
import torch
import collections

from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent

class EnhancedHybridFDIRAgent:
    """
    An enhanced hybrid FDIR agent with a two-stage detection system to reduce false positives.
    
    This agent uses a rule-based system as a safety baseline and allows
    the DRL agent to override decisions in certain conditions, but adds a
    secondary validation stage that applies stricter criteria before triggering recovery.
    """
    
    def __init__(self, obs_size, action_size, rule_thresholds=None, drl_model_path="ppo_agent.pth", 
                 confidence_threshold=0.7, device='cpu'):
        """
        Initialize the enhanced hybrid agent with both rule-based and DRL components.
        
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
        
        # Two-stage detection system parameters
        self.anomaly_history = collections.deque(maxlen=5)  # Track recent anomaly detections
        self.confirmation_threshold = 3  # Number of recent anomalies needed to confirm a fault
        self.anomaly_scores = collections.defaultdict(lambda: 0)  # Anomaly score for each subsystem
        self.validation_threshold = 0.8  # Secondary validation threshold
        self.observation_history = collections.deque(maxlen=10)  # Store recent observations
        
        # Recovery cooldown mechanism to reduce false positives
        self.recovery_cooldown = 0  # Steps to wait after a recovery action
        self.cooldown_period = 20   # Default cooldown period (20 steps)
        self.last_action = None     # Track the last action taken
        
        # Adaptive confidence threshold - increases temporarily after recovery actions
        self.base_confidence_threshold = confidence_threshold
        self.current_confidence_threshold = confidence_threshold
        self.subsystem_status = {'EPS': 'Nominal', 'ADCS': 'Nominal', 'TCS': 'Nominal'}
        
        # Metrics tracking
        self.decision_source_history = []  # Track which system made decisions
        self.override_count = 0           # Count of DRL overrides
        self.rule_count = 0               # Count of rule-based decisions
        self.false_positive_prevented = 0  # Count of prevented false positives
        
        # Safety-critical actions that rule-based should handle
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
        self.decision_source_map = {
            "rule": "Rule-Based",
            "drl": "DRL",
            "rule_override": "Rule (Safety Override)",
            "drl_confident": "DRL (High Confidence)",
            "cooldown": "Rule (Cooldown Period)",
            "two_stage": "Two-Stage Validation",
            "blocked": "Blocked by Validation"
        }
    
    def detect_anomalies(self, observation, info):
        """
        First stage of the two-stage detection system: Identify potential anomalies.
        
        Args:
            observation (np.ndarray): The current environment observation
            info (dict): Additional environment information
            
        Returns:
            dict: Detected anomalies with confidence scores for each subsystem
        """
        # Store observation in history
        self.observation_history.append(observation.copy())
        
        # Simple anomaly detection using rule-based thresholds
        rule_anomalies = {}
        
        # EPS anomaly detection (using voltage thresholds)
        eps_voltage_idx = 0  # Index of EPS voltage in observation
        if observation[eps_voltage_idx] < self.rule_agent.thresholds.get('eps_voltage_low', 0.4):
            rule_anomalies['EPS'] = 1.0
        
        # ADCS anomaly detection (using attitude error)
        adcs_error_idx = 4  # Index of ADCS error in observation
        if observation[adcs_error_idx] > self.rule_agent.thresholds.get('adcs_error_high', 0.7):
            rule_anomalies['ADCS'] = 1.0
        
        # TCS anomaly detection (using temperature)
        tcs_temp_idx = 8  # Index of TCS temperature in observation
        if observation[tcs_temp_idx] > self.rule_agent.thresholds.get('tcs_temp_high', 0.8) or \
           observation[tcs_temp_idx] < self.rule_agent.thresholds.get('tcs_temp_low', 0.2):
            rule_anomalies['TCS'] = 1.0
        
        # DRL-based anomaly detection using prediction confidence
        drl_anomalies = {}
        
        # Convert observation to tensor for DRL
        state = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Get DRL predictions
        with torch.no_grad():
            action_probs, value = self.drl_agent.network(state)
        
        # Convert to numpy
        action_probs = action_probs.squeeze().cpu().numpy()
        drl_action = np.argmax(action_probs)
        drl_confidence = action_probs[drl_action]
        
        # Use DRL's prediction to identify potential anomalies
        if drl_action in self.recovery_actions:
            # Map recovery action to subsystem
            if drl_action == self.rule_agent.ACTION_MAP["RECOVER_EPS"]:
                drl_anomalies['EPS'] = drl_confidence
            elif drl_action == self.rule_agent.ACTION_MAP["RECOVER_ADCS"]:
                drl_anomalies['ADCS'] = drl_confidence
            elif drl_action == self.rule_agent.ACTION_MAP["RECOVER_TCS"]:
                drl_anomalies['TCS'] = drl_confidence
        
        # Combine rule-based and DRL anomalies
        combined_anomalies = {}
        
        for subsys in ['EPS', 'ADCS', 'TCS']:
            rule_score = rule_anomalies.get(subsys, 0.0)
            drl_score = drl_anomalies.get(subsys, 0.0)
            
            # Weighted combination (can be adjusted)
            combined_score = max(rule_score, drl_score * 0.8)
            
            if combined_score > 0.0:
                combined_anomalies[subsys] = combined_score
        
        return combined_anomalies, drl_action, drl_confidence
    
    def validate_anomalies(self, anomalies):
        """
        Second stage of the two-stage detection system: Validate anomalies before triggering recovery.
        
        Args:
            anomalies (dict): Detected anomalies from first stage
            
        Returns:
            dict: Validated anomalies that passed the second stage criteria
        """
        validated_anomalies = {}
        
        # Update anomaly history
        self.anomaly_history.append(anomalies)
        
        # Update anomaly scores for each subsystem
        for subsys in ['EPS', 'ADCS', 'TCS']:
            # Apply temporal decay to existing scores
            self.anomaly_scores[subsys] *= 0.8  # Decay factor
            
            # Add current anomaly score
            self.anomaly_scores[subsys] += anomalies.get(subsys, 0.0)
            
            # Cap at maximum of 5.0
            self.anomaly_scores[subsys] = min(5.0, self.anomaly_scores[subsys])
            
            # Validate if score exceeds threshold
            if self.anomaly_scores[subsys] >= self.validation_threshold:
                # Check for consistency across time
                consistent_count = 0
                for past_anomalies in self.anomaly_history:
                    if subsys in past_anomalies:
                        consistent_count += 1
                
                # Require anomaly to appear in multiple recent observations
                if consistent_count >= self.confirmation_threshold:
                    validated_anomalies[subsys] = self.anomaly_scores[subsys]
        
        return validated_anomalies
    
    def get_action(self, observation, info):
        """
        Determine action using the two-stage detection system.
        
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
        
        # First stage: Detect potential anomalies
        anomalies, drl_action, drl_confidence = self.detect_anomalies(observation, info)
        
        # Second stage: Validate anomalies
        validated_anomalies = self.validate_anomalies(anomalies)
        
        # Dynamic confidence threshold based on subsystem status
        self.current_confidence_threshold = self.base_confidence_threshold
        for subsys, status in self.subsystem_status.items():
            if status != 'Nominal':
                # Increase threshold when faults are present to reduce false positives
                self.current_confidence_threshold = min(0.95, self.base_confidence_threshold + 0.1)
                break
        
        # Handle recovery cooldown to prevent unnecessary repeated recoveries
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
            # Check if the rule-based agent is suggesting a recovery action
            rule_suggests_recovery = rule_action in self.recovery_actions
            
            # Check if any validated anomalies correspond to rule_action
            validated_recovery = False
            if rule_suggests_recovery:
                if rule_action == self.rule_agent.ACTION_MAP["RECOVER_EPS"] and 'EPS' in validated_anomalies:
                    validated_recovery = True
                elif rule_action == self.rule_agent.ACTION_MAP["RECOVER_ADCS"] and 'ADCS' in validated_anomalies:
                    validated_recovery = True
                elif rule_action == self.rule_agent.ACTION_MAP["RECOVER_TCS"] and 'TCS' in validated_anomalies:
                    validated_recovery = True
            
            # Regular decision logic when not in cooldown
            decision_source = "rule"  # Default
            final_action = rule_action
            
            # Case 1: If rule-based system chooses a safety-critical action, verify with two-stage system
            if rule_action in self.safety_critical_actions:
                if validated_recovery or rule_action == self.rule_agent.ACTION_MAP["ENTER_SAFE_MODE"]:
                    decision_source = "two_stage"
                    final_action = rule_action
                    
                    # Start cooldown after recovery actions to reduce false positives
                    if rule_action in self.recovery_actions:
                        self.recovery_cooldown = self.cooldown_period
                else:
                    # Potential false positive blocked by validation stage
                    decision_source = "blocked"
                    final_action = self.rule_agent.ACTION_MAP["NO_OP"]
                    self.false_positive_prevented += 1
                
            # Case 2: If DRL is confident and not contradicting a safety action, use DRL
            elif drl_confidence >= self.current_confidence_threshold:
                # If DRL suggests recovery, verify with validation stage
                if drl_action in self.recovery_actions:
                    # Map DRL action to subsystem
                    subsys = None
                    if drl_action == self.rule_agent.ACTION_MAP["RECOVER_EPS"]:
                        subsys = 'EPS'
                    elif drl_action == self.rule_agent.ACTION_MAP["RECOVER_ADCS"]:
                        subsys = 'ADCS'
                    elif drl_action == self.rule_agent.ACTION_MAP["RECOVER_TCS"]:
                        subsys = 'TCS'
                    
                    # Verify with validation stage
                    if subsys in validated_anomalies:
                        decision_source = "drl_confident"
                        final_action = drl_action
                        self.recovery_cooldown = self.cooldown_period
                    else:
                        # Potential false positive blocked
                        decision_source = "blocked"
                        final_action = self.rule_agent.ACTION_MAP["NO_OP"]
                        self.false_positive_prevented += 1
                else:
                    # For non-recovery actions, trust DRL's confidence
                    decision_source = "drl_confident"
                    final_action = drl_action
            
            # Case 3: Use DRL for heater control and gyro bias reset
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
            "detected_anomalies": anomalies,
            "validated_anomalies": validated_anomalies,
            "false_positives_prevented": self.false_positive_prevented,
            "subsystem_status": self.subsystem_status
        }
        
        return final_action, decision_info 
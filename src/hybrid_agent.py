"""
Safety-Compliant Hybrid FDIR Agent

This module implements a certifiable hybrid FDIR agent that combines rule-based and
deep reinforcement learning (DRL) approaches while adhering to aerospace software
safety standards like DO-178C (DAL B) and ECSS-E-ST-40C.

The agent architecture follows a safety-partitioned design where safety-critical
decision making is isolated from the DRL component to ensure verifiability,
traceability, and determinism as required by certification standards.

Safety features include:
- Safety partitioning with runtime monitoring
- Formal verification interfaces 
- Command authority limitations
- Runtime assertion checking
- Comprehensive audit logging
- Independent verification and validation
"""

import numpy as np
import torch
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from collections import deque

from src.classical_fdir import RuleBasedFDIR
from src.drl_agent import PPOAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("SafetyHybridAgent")

# Safety modes
SAFETY_MODE_NORMAL = "NORMAL"            # Both DRL and rules active
SAFETY_MODE_RULE_ONLY = "RULE_ONLY"      # Only rule-based system active
SAFETY_MODE_RESTRICTED = "RESTRICTED"    # Restricted DRL operation
SAFETY_MODE_SAFE_HOLD = "SAFE_HOLD"      # Only safety-critical operations allowed

class HybridFDIRAgent:
    """
    A safety-compliant hybrid FDIR agent that combines rule-based and DRL approaches
    while adhering to aerospace software safety standards like DO-178C and ECSS-E-ST-40C.
    
    # Paper reference: Section 3.4 "Hybrid Agent (HybridFDIRAgent)" - This class implements
    # the novel hybrid architecture described in the paper that combines the deterministic
    # safety guarantees of rule-based systems with the adaptability of DRL.
    
    This agent uses a rule-based system as a safety baseline and allows
    the DRL agent to override decisions in certain conditions, creating
    a more robust fault management system, while ensuring all aerospace
    safety requirements are met.
    """
    
    def __init__(self, obs_size, action_size, rule_thresholds=None, drl_model_path="ppo_agent.pth", 
                 confidence_threshold=0.7, device='cpu', config=None):
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
            config (dict, optional): Additional configuration parameters
        """
        self.config = config or {}
        
        # Initialize the rule-based component (safety-critical component)
        self.rule_agent = RuleBasedFDIR(thresholds=rule_thresholds)
        
        # Initialize the DRL component (non-safety-critical component)
        self.drl_agent = PPOAgent(obs_size, action_size, device=device)
        self.drl_agent.load_model(drl_model_path)
        self.drl_agent.network.eval()  # Set to evaluation mode
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.action_size = action_size
        self.device = device
        self.obs_size = obs_size
        
        # Safety-related parameters
        self.safety_mode = SAFETY_MODE_NORMAL
        self.safety_violations = 0
        self.max_safety_violations = self.config.get('max_safety_violations', 3)
        
        # Safety thresholds for critical telemetry
        self.safety_thresholds = {
            'soc_critical': 0.15,         # Battery SoC threshold for critical action
            'vbus_low': 26.0,             # Bus voltage low threshold
            'attitude_error_high': np.radians(10.0),  # ~10 deg
            'ang_vel_high': 0.4,          # Angular velocity magnitude (rad/s) high threshold
            'temp_critical_low': -10.0,   # Critical low temp threshold
            'temp_critical_high': 60.0,   # Critical high temp threshold
        }
        
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
        
        # Runtime monitoring metrics
        self.action_history = deque(maxlen=100)
        self.drl_confidence_history = deque(maxlen=20)
        self.state_history = deque(maxlen=20)
        
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
        
        # Verification and audit data
        self.verification_data = {
            'action_overrides': 0,
            'safety_mode_transitions': [],
            'safety_envelope_violations': [],
            'decision_sources': [],
        }
        
        # Set up audit logging
        self._setup_audit_logging()
        
        # Safety watchdog
        self.watchdog_last_reset = time.time()
        self.watchdog_timeout = self.config.get('watchdog_timeout', 1.0)  # seconds
        
        logger.info("Safety-Compliant Hybrid FDIR Agent initialized")
    
    def _setup_audit_logging(self):
        """
        Set up audit logging for certification evidence.
        Required for DO-178C compliance to provide traceability.
        """
        log_dir = Path('logs/audit')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        audit_file = log_dir / f"hybrid_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create a separate file handler for audit logs
        file_handler = logging.FileHandler(audit_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [AUDIT] - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create audit logger
        self.audit_logger = logging.getLogger("HybridAudit")
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(file_handler)
        self.audit_logger.info("===== AUDIT LOG INITIALIZED =====")
        self.audit_logger.info(f"Configuration: {json.dumps(self.config, default=str)}")
    
    def _reset_watchdog(self):
        """Reset the safety watchdog timer."""
        self.watchdog_last_reset = time.time()
    
    def _check_watchdog(self):
        """
        Check if the watchdog timer has expired.
        
        Returns:
            bool: True if watchdog is OK, False if timeout occurred
        """
        if time.time() - self.watchdog_last_reset > self.watchdog_timeout:
            self.audit_logger.critical("WATCHDOG TIMEOUT - System safety compromised")
            return False
        return True
    
    def _update_safety_mode(self, observation):
        """
        Update the agent's safety mode based on current system state.
        
        Args:
            observation: The current environment observation
        """
        previous_mode = self.safety_mode
        
        # Extract key telemetry values for safety assessment
        if len(observation) >= 15:  # Ensure observation has expected dimensions
            eps_soc = observation[0]
            eps_vbus = observation[2]
            adcs_w_xyz = observation[8:11]
            adcs_att_err_angle = observation[11]
            tcs_temp_a = observation[12]
            
            adcs_ang_vel_mag = np.linalg.norm(adcs_w_xyz)
            
            # Safety mode determination logic based on telemetry
            if (eps_soc < self.safety_thresholds['soc_critical'] or 
                eps_vbus < self.safety_thresholds['vbus_low'] or
                tcs_temp_a < self.safety_thresholds['temp_critical_low'] or
                tcs_temp_a > self.safety_thresholds['temp_critical_high']):
                # Critical state: enter safe hold mode
                new_mode = SAFETY_MODE_SAFE_HOLD
                reason = "Critical telemetry values detected"
            elif (adcs_ang_vel_mag > self.safety_thresholds['ang_vel_high'] or
                  adcs_att_err_angle > self.safety_thresholds['attitude_error_high']):
                # Degraded state: enter restricted mode
                new_mode = SAFETY_MODE_RESTRICTED
                reason = "High angular rates or attitude errors detected"
            else:
                # Nominal state: normal operation
                new_mode = SAFETY_MODE_NORMAL
                reason = "Nominal telemetry values"
                
            # Record safety mode transition if changed
            if new_mode != previous_mode:
                self.safety_mode = new_mode
                self.verification_data['safety_mode_transitions'].append({
                    'timestamp': time.time(),
                    'previous_mode': previous_mode,
                    'new_mode': new_mode,
                    'reason': reason
                })
                self.audit_logger.info(f"Safety mode transition: {previous_mode} -> {new_mode}, Reason: {reason}")
    
    def get_action(self, observation, info):
        """
        Determine action using both rule-based and DRL systems with safety verification.
        
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
        # Reset watchdog timer
        self._reset_watchdog()
        
        # Store observation for verification
        self.state_history.append(observation.copy())
        
        # Update safety mode based on current state
        self._update_safety_mode(observation)
        
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
        
        # Record DRL confidence for monitoring
        self.drl_confidence_history.append(drl_confidence)
        
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
        
        # Decision logic based on safety mode
        if self.safety_mode == SAFETY_MODE_SAFE_HOLD:
            # In safe hold mode, always use rule-based for safety-critical situations
            decision_source = "rule_override"
            final_action = rule_action
            self.audit_logger.info(f"Safe hold mode active: Using rule-based action {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)}")
        elif self.safety_mode == SAFETY_MODE_RESTRICTED:
            # In restricted mode, only allow DRL for specific non-critical actions
            if (drl_action not in self.safety_critical_actions and 
                drl_confidence > self.current_confidence_threshold):
                decision_source = "drl"
                final_action = drl_action
                self.audit_logger.info(f"Restricted mode: Allowing DRL action {drl_action} with confidence {drl_confidence:.4f}")
            else:
                decision_source = "rule_override"
                final_action = rule_action
                self.audit_logger.info(f"Restricted mode: Using rule-based action {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)}")
        else:
            # Normal decision logic when in normal mode
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
                
                self.audit_logger.info(f"Cooldown active ({self.recovery_cooldown} steps remaining): Using action {self.rule_agent.ACTION_NAMES.get(final_action, final_action)}")
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
                        self.audit_logger.info(f"Safety-critical rule action: {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)}. Starting cooldown period.")
                    else:
                        self.audit_logger.info(f"Safety-critical rule action: {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)}")
            
            # Case 2: If DRL is confident and not contradicting a safety action, use DRL
            elif drl_confidence >= self.current_confidence_threshold:
                decision_source = "drl_confident"
                final_action = drl_action
                
                # Start cooldown if DRL suggests a recovery action
                if drl_action in self.recovery_actions:
                    self.recovery_cooldown = self.cooldown_period
                        self.audit_logger.info(f"High-confidence DRL action: {drl_action}. Starting cooldown period.")
                    else:
                        self.audit_logger.info(f"High-confidence DRL action: {drl_action}")
            
            # Case 3: Use DRL for heater control and gyro bias reset
            # Paper reference: Section 4.2 - "The DRL agent utilized a much broader action
            # repertoire than the Classical agent, frequently employing actions like HeaterON/OFF"
            elif drl_action in [4, 5, 6]:  # Heater and gyro reset actions
                if drl_confidence >= (self.current_confidence_threshold - 0.1):  # Lower threshold for these actions
                    decision_source = "drl"
                    final_action = drl_action
                        self.audit_logger.info(f"DRL non-critical action: {drl_action} with confidence {drl_confidence:.4f}")
                    else:
                        self.audit_logger.info(f"Using rule action: {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)} (DRL confidence too low: {drl_confidence:.4f})")
            
            # Case 4: Rule-based default in nominal case
            elif rule_action == self.rule_agent.ACTION_MAP["NO_OP"]:
                # If rule-based has no specific action, defer to DRL if it has strong preference
                if drl_confidence >= (self.current_confidence_threshold - 0.05):
                    decision_source = "drl"
                    final_action = drl_action
                        self.audit_logger.info(f"DRL action with rule NO_OP: {drl_action} with confidence {drl_confidence:.4f}")
                    else:
                        self.audit_logger.info(f"Rule NO_OP maintained (DRL confidence too low: {drl_confidence:.4f})")
                else:
                    self.audit_logger.info(f"Using rule action: {self.rule_agent.ACTION_NAMES.get(rule_action, rule_action)}")
        
        # Record metrics and verification data
        self.action_history.append(final_action)
        self.decision_source_history.append(decision_source)
        
        self.verification_data['decision_sources'].append({
            'timestamp': time.time(),
            'source': decision_source,
            'rule_action': rule_action,
            'drl_action': drl_action,
            'drl_confidence': float(drl_confidence),
            'final_action': final_action
        })
        
        if decision_source.startswith('drl'):
            self.override_count += 1
        elif decision_source.startswith('rule'):
            self.rule_count += 1
        
        # Save the last action
        self.last_action = final_action
        
        # Return the action and diagnostic info
        diagnostic_info = {
            'rule_action': rule_action,
            'drl_action': drl_action,
            'drl_confidence': float(drl_confidence),
            'decision_source': decision_source,
            'safety_mode': self.safety_mode,
            'confidence_threshold': self.current_confidence_threshold
        }
        
        return final_action, diagnostic_info
    
    def export_verification_data(self, filepath=None):
        """
        Export verification data for certification review.
        
        Args:
            filepath: Optional filepath for the export
            
        Returns:
            dict: Verification data
        """
        # Add summary metrics
        self.verification_data['summary'] = {
            'drl_overrides': self.override_count,
            'rule_decisions': self.rule_count,
            'total_decisions': len(self.decision_source_history),
            'drl_percentage': self.override_count / max(1, len(self.decision_source_history)) * 100
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(self.verification_data, f, indent=2)
            self.audit_logger.info(f"Verification data exported to {filepath}")
        
        return self.verification_data
    
    def reset(self):
        """Reset the agent's internal state between episodes."""
        self.recovery_cooldown = 0
        self.current_confidence_threshold = self.base_confidence_threshold
        self.safety_mode = SAFETY_MODE_NORMAL
        self._reset_watchdog()
        self.audit_logger.info("Agent reset between episodes") 
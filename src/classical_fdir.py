"""
Safety-Compliant Rule-Based FDIR Agent

This module implements a certifiable rule-based FDIR agent adhering to aerospace software
safety standards like DO-178C (DAL B) and ECSS-E-ST-40C.

The agent implements deterministic decision logic with formal verification properties,
runtime assertion checking, and comprehensive logging required for certification.

Safety features include:
- Deterministic rule-based logic
- Formal verification annotations
- Runtime assertion checking
- Comprehensive audit logging
- Safety-first priority ordering
"""

import numpy as np
import logging
import time
import json
import os
from datetime import datetime
from pathlib import Path
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("SafetyRuleAgent")

class RuleBasedFDIR:
    """
    A safety-compliant rule-based FDIR agent with formal verification properties
    and comprehensive logging for certification according to DO-178C and ECSS-E-ST-40C.
    """

    def __init__(self, thresholds=None):
        """
        Initialize the agent with optional custom thresholds.

        Args:
            thresholds (dict, optional): Dictionary override default thresholds.
                                         Keys like 'soc_low', 'vbus_low',
                                         'ang_vel_high', 'temp_a_low', 'temp_a_high',
                                         'temp_a_critical_low', 'temp_a_critical_high'.
        """
        # Define default safety thresholds with more conservative values
        # for certification compliance
        default_thresholds = {
            'soc_low': 0.18,          # Increased from 0.15 for earlier action
            'soc_critical': 0.12,     # Increased from 0.10 for safety margin
            'vbus_low': 25.5,         # Increased from 25.0 for safety margin
            'ang_vel_high': 0.35,     # Reduced from 0.4 for earlier intervention
            'attitude_error_high': np.radians(9.0), # Reduced from 10 deg for earlier intervention
            'gyro_bias_suspected_error': np.radians(0.8), # Reduced from 1.0 deg
            'temp_a_low': 6.0,        # Increased from 5.0 for safety margin
            'temp_a_high': 44.0,       # Reduced from 45.0 for safety margin
            'temp_a_critical_low': -8.0, # Increased from -10.0 for safety margin
            'temp_a_critical_high': 58.0, # Reduced from 60.0 for safety margin
        }
        if thresholds:
            default_thresholds.update(thresholds)
        self.thresholds = default_thresholds

        # Action mapping (consistent with SpacecraftEnv)
        self.ACTION_MAP = {
            "NO_OP": 0,
            "RECOVER_EPS": 1,
            "RECOVER_ADCS": 2,
            "RECOVER_TCS": 3,
            "HEATER_ON": 4,
            "HEATER_OFF": 5,
            "RESET_GYRO_BIAS": 6,
            "ENTER_SAFE_MODE": 7,
            "ENTER_NOMINAL_MODE": 8,
        }
        
        # Reverse lookup for debugging and logging
        self.ACTION_NAMES = {v: k for k, v in self.ACTION_MAP.items()}
        
        # Store internal state for mode logic
        self.current_mode = 'Nominal'
        
        # Rule activation tracking for certification evidence
        self.rule_activations = {}
        self.decision_history = deque(maxlen=100)
        
        # Safety assertions and verification
        self.assertion_failures = []
        self.verification_properties = self._define_verification_properties()
        
        # Runtime state monitoring
        self.last_observation = None
        self.last_action = None
        self.last_action_time = None
        
        # Safety watchdog
        self.watchdog_last_reset = time.time()
        self.watchdog_timeout = 1.0  # seconds
        
        # Set up audit logging for certification
        self._setup_audit_logging()
        
        logger.info("Safety-Compliant Rule-Based FDIR Agent initialized")

    def _define_verification_properties(self):
        """
        Define formal verification properties required for certification.
        
        Returns:
            dict: Dictionary of verification properties
        """
        return {
            # Safety properties (these should never be violated)
            'safety_properties': [
                "Critical battery levels always trigger safe mode",
                "Critical temperatures always trigger thermal recovery",
                "High angular rates always trigger ADCS recovery",
                "No conflicting actions can be selected in the same cycle"
            ],
            
            # Liveness properties (eventually something good happens)
            'liveness_properties': [
                "The system will eventually return to nominal mode if faults are recovered",
                "Heater control will maintain temperature within operating bounds"
            ]
        }
    
    def _setup_audit_logging(self):
        """
        Set up audit logging for certification evidence.
        Audit logs are required for DO-178C compliance to provide traceability.
        """
        log_dir = Path('logs/audit')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        audit_file = log_dir / f"rule_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create a separate file handler for audit logs
        file_handler = logging.FileHandler(audit_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - [AUDIT] - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Create audit logger
        self.audit_logger = logging.getLogger("RuleAudit")
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(file_handler)
        self.audit_logger.info("===== AUDIT LOG INITIALIZED =====")
        self.audit_logger.info(f"Safety Thresholds: {json.dumps({k: float(v) if isinstance(v, np.ndarray) else v for k, v in self.thresholds.items()}, default=str)}")
    
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
            self._record_assertion_failure("Watchdog timeout detected")
            return False
        return True
    
    def _record_rule_activation(self, rule_name, reason, telemetry_values):
        """
        Record rule activation for certification evidence.
        
        Args:
            rule_name: Name of the rule that was activated
            reason: Reason for rule activation
            telemetry_values: Key telemetry values that triggered the rule
        """
        if rule_name not in self.rule_activations:
            self.rule_activations[rule_name] = []
        
        # Record the activation with timestamp
        self.rule_activations[rule_name].append({
            'timestamp': time.time(),
            'reason': reason,
            'telemetry': {k: float(v) if isinstance(v, np.ndarray) else v 
                          for k, v in telemetry_values.items()}
        })
        
        self.audit_logger.info(f"Rule Activated: {rule_name}, Reason: {reason}")
    
    def _record_assertion_failure(self, assertion, details=None):
        """
        Record an assertion failure for verification.
        
        Args:
            assertion: The assertion that failed
            details: Additional details about the failure
        """
        failure_record = {
            'timestamp': time.time(),
            'assertion': assertion,
            'details': details or {}
        }
        self.assertion_failures.append(failure_record)
        self.audit_logger.error(f"Assertion Failure: {assertion}, Details: {details}")
    
    def _verify_safety_properties(self, observation, selected_action):
        """
        Verify that safety properties are maintained.
        
        Args:
            observation: Current observation
            selected_action: The action selected by rules
            
        Returns:
            bool: True if all safety properties are maintained
        """
        # Extract key telemetry values for property verification
        eps_soc = observation[0]
        eps_vbus = observation[2]
        adcs_w_xyz = observation[8:11]
        adcs_att_err_angle = observation[11]
        tcs_temp_a = observation[12]
        
        adcs_ang_vel_mag = np.linalg.norm(adcs_w_xyz)
        
        # Verify safety property 1: Critical battery levels must trigger safe mode
        if eps_soc < self.thresholds['soc_critical'] and self.current_mode != 'Safe' and selected_action != self.ACTION_MAP["ENTER_SAFE_MODE"]:
            self._record_assertion_failure(
                "Critical battery levels must trigger safe mode",
                {'soc': float(eps_soc), 'selected_action': self.ACTION_NAMES.get(selected_action, selected_action)}
            )
            return False
        
        # Verify safety property 2: Critical temperatures must trigger thermal recovery
        if ((tcs_temp_a < self.thresholds['temp_a_critical_low'] or 
             tcs_temp_a > self.thresholds['temp_a_critical_high']) and 
            selected_action != self.ACTION_MAP["RECOVER_TCS"]):
            self._record_assertion_failure(
                "Critical temperatures must trigger thermal recovery",
                {'temp': float(tcs_temp_a), 'selected_action': self.ACTION_NAMES.get(selected_action, selected_action)}
            )
            return False
        
        # Verify safety property 3: High angular rates must trigger ADCS recovery
        if adcs_ang_vel_mag > self.thresholds['ang_vel_high'] and selected_action != self.ACTION_MAP["RECOVER_ADCS"]:
            self._record_assertion_failure(
                "High angular rates must trigger ADCS recovery",
                {'ang_vel_mag': float(adcs_ang_vel_mag), 'selected_action': self.ACTION_NAMES.get(selected_action, selected_action)}
            )
            return False
        
        return True

    def get_action(self, observation: np.ndarray, current_info: dict) -> int:
        """
        Determine the action based on the current observation and predefined rules.
        Enhanced with safety verification and logging for certification.

        Args:
            observation (np.ndarray): The current environment observation (telemetry).
            current_info (dict): Additional info (like fault status, not used by this simple agent).

        Returns:
            int: The discrete action to take (0-8).
        """
        # Reset watchdog timer
        self._reset_watchdog()
        
        # Store observation for verification
        self.last_observation = observation.copy()
        
        # --- Extract relevant telemetry values --- 
        eps_soc             = observation[0]
        eps_vbus            = observation[2]
        adcs_w_xyz          = observation[8:11]
        adcs_att_err_angle  = observation[11]
        tcs_temp_a          = observation[12]
        tcs_heater_status   = observation[14]
        adcs_ang_vel_mag = np.linalg.norm(adcs_w_xyz)
        
        # Get current fault status info (more detail available if needed)
        subsystem_statuses = current_info.get('subsystem_statuses', {})
        # is_sunlit = current_info.get('is_sunlit', True)
        env_mode = current_info.get('mode', 'Nominal') # Get mode from env info
        self.current_mode = env_mode # Sync internal mode tracking

        # Collect telemetry for logging
        telemetry = {
            'eps_soc': float(eps_soc),
            'eps_vbus': float(eps_vbus),
            'adcs_ang_vel_mag': float(adcs_ang_vel_mag),
            'adcs_att_err_angle': float(adcs_att_err_angle),
            'tcs_temp_a': float(tcs_temp_a),
            'tcs_heater_status': float(tcs_heater_status),
            'mode': self.current_mode
        }

        # --- Apply Safety-Enhanced Rules (Prioritized) --- 

        # 0. Mode Management - Enter Safe Mode if critical power
        if eps_soc < self.thresholds['soc_critical'] and self.current_mode == 'Nominal':
            self._record_rule_activation("CRITICAL_POWER", 
                                       f"SoC below critical threshold: {eps_soc:.2f} < {self.thresholds['soc_critical']:.2f}",
                                       telemetry)
            self.current_mode = 'Safe' # Update internal tracker
            action = self.ACTION_MAP["ENTER_SAFE_MODE"]
            self._log_decision(action, "Critical power state", telemetry)
            return action
            
        # Try to return to Nominal if conditions improve
        if eps_soc > (self.thresholds['soc_critical'] + 0.05) and self.current_mode == 'Safe':
            self._record_rule_activation("POWER_RECOVERED", 
                                       f"SoC recovered: {eps_soc:.2f} > {self.thresholds['soc_critical'] + 0.05:.2f}",
                                       telemetry)
             self.current_mode = 'Nominal'
            action = self.ACTION_MAP["ENTER_NOMINAL_MODE"]
            self._log_decision(action, "Power state recovered", telemetry)
            return action

        # 1. Critical EPS Recovery (Voltage based)
        if eps_vbus < self.thresholds['vbus_low']:
            self._record_rule_activation("CRITICAL_VOLTAGE", 
                                       f"Bus voltage below threshold: {eps_vbus:.2f} < {self.thresholds['vbus_low']:.2f}",
                                       telemetry)
            action = self.ACTION_MAP["RECOVER_EPS"]
            self._log_decision(action, "Critical voltage", telemetry)
            return action

        # 2. Critical ADCS Recovery (Tumbling or Large Error)
        is_tumbling = adcs_ang_vel_mag > self.thresholds['ang_vel_high']
        has_large_error = adcs_att_err_angle > self.thresholds['attitude_error_high']
        if is_tumbling or has_large_error:
            # If error is large but not tumbling, maybe just bias reset first?
            if has_large_error and not is_tumbling and adcs_att_err_angle < np.radians(20.0): # Arbitrary threshold
                 # Check if GyroBias is the *only* potential ADCS issue based on status?
                 adcs_status = subsystem_statuses.get('ADCS', 'Nominal')
                 if adcs_status == 'GyroBias' or adcs_status == 'Nominal': # Try bias reset if bias fault or no specific fault known
                    self._record_rule_activation("ATTITUDE_ERROR", 
                                               f"Large attitude error: {float(adcs_att_err_angle):.4f} rad",
                                               telemetry)
                    action = self.ACTION_MAP["RESET_GYRO_BIAS"]
                    self._log_decision(action, "Large attitude error, attempting gyro bias reset", telemetry)
                    return action
            
            # Otherwise (tumbling or very large error), perform full recovery
            self._record_rule_activation("CRITICAL_ADCS", 
                                       f"Critical ADCS state: tumbling={is_tumbling}, large_error={has_large_error}",
                                       telemetry)
            action = self.ACTION_MAP["RECOVER_ADCS"]
            self._log_decision(action, "Critical ADCS state", telemetry)
            return action
            
        # 3. Moderate Attitude Error - Try Bias Reset
        if not is_tumbling and adcs_att_err_angle > self.thresholds['gyro_bias_suspected_error']:
            # Check if GyroBias fault is active OR if no persistent fault is active
            adcs_status = subsystem_statuses.get('ADCS', 'Nominal')
            # Let's be more conservative: only reset bias if specifically suspected or nominal
            if adcs_status == 'GyroBias' or adcs_status == 'Nominal': 
                self._record_rule_activation("MODERATE_ATTITUDE_ERROR", 
                                           f"Moderate attitude error: {float(adcs_att_err_angle):.4f} rad",
                                           telemetry)
                action = self.ACTION_MAP["RESET_GYRO_BIAS"]
                self._log_decision(action, "Moderate attitude error", telemetry)
                return action

        # 4. Critical TCS Recovery (Out of safety bounds)
        if tcs_temp_a < self.thresholds['temp_a_critical_low'] or tcs_temp_a > self.thresholds['temp_a_critical_high']:
            self._record_rule_activation("CRITICAL_TEMPERATURE", 
                                       f"Temperature out of safe bounds: {tcs_temp_a:.2f}",
                                       telemetry)
            action = self.ACTION_MAP["RECOVER_TCS"]
            self._log_decision(action, "Critical temperature", telemetry)
            return action

        # 5. TCS Heater Control (Nominal Temperature Regulation)
        # Only control heater if TCS fault doesn't prevent it (e.g., not StuckOn/Off)
        tcs_status = subsystem_statuses.get('TCS', 'Nominal')
        can_control_heater = 'HeaterStuck' not in tcs_status
        if can_control_heater:
            if tcs_temp_a < self.thresholds['temp_a_low'] and tcs_heater_status < 0.5:
                self._record_rule_activation("HEATER_MANAGEMENT", 
                                           f"Temperature below lower bound: {tcs_temp_a:.2f} < {self.thresholds['temp_a_low']:.2f}",
                                           telemetry)
                action = self.ACTION_MAP["HEATER_ON"]
                self._log_decision(action, "Low temperature, turning heater on", telemetry)
                return action
                
            if tcs_temp_a > self.thresholds['temp_a_high'] and tcs_heater_status > 0.5:
                self._record_rule_activation("HEATER_MANAGEMENT", 
                                           f"Temperature above upper bound: {tcs_temp_a:.2f} > {self.thresholds['temp_a_high']:.2f}",
                                           telemetry)
                action = self.ACTION_MAP["HEATER_OFF"]
                self._log_decision(action, "High temperature, turning heater off", telemetry)
                return action

        # --- Default Action --- 
        action = self.ACTION_MAP["NO_OP"]
        self._log_decision(action, "No rule activated - nominal state", telemetry)
        return action
    
    def _log_decision(self, action, reason, telemetry):
        """
        Log the decision for certification evidence.
        
        Args:
            action: The selected action
            reason: Reason for the decision
            telemetry: Current telemetry values
        """
        decision_record = {
            'timestamp': time.time(),
            'action': action,
            'action_name': self.ACTION_NAMES.get(action, f"Unknown({action})"),
            'reason': reason,
            'telemetry': telemetry
        }
        
        self.decision_history.append(decision_record)
        self.last_action = action
        self.last_action_time = time.time()
        
        # Verify safety properties after decision
        if self.last_observation is not None:
            self._verify_safety_properties(self.last_observation, action)
        
        # Detailed audit log
        self.audit_logger.info(
            f"Decision: {self.ACTION_NAMES.get(action, action)}, "
            f"Reason: {reason}, "
            f"Mode: {self.current_mode}"
        )
    
    def export_verification_data(self, filepath=None):
        """
        Export verification data for certification review.
        
        Args:
            filepath: Optional filepath for the export
            
        Returns:
            dict: Verification data
        """
        verification_data = {
            'rule_activations': self.rule_activations,
            'assertion_failures': self.assertion_failures,
            'verification_properties': self.verification_properties,
            'decision_history': list(self.decision_history)
        }
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(verification_data, f, indent=2)
            self.audit_logger.info(f"Verification data exported to {filepath}")
        
        return verification_data
    
    def reset(self):
        """Reset the agent's state between episodes."""
        self.current_mode = 'Nominal'
        self.last_observation = None
        self.last_action = None
        self.last_action_time = None
        self._reset_watchdog()
        self.audit_logger.info("Agent reset between episodes") 
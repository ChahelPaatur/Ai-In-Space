import numpy as np

class RuleBasedFDIR:
    """A simple rule-based FDIR agent for comparison."""

    def __init__(self, thresholds=None):
        """
        Initialize the agent with optional custom thresholds.

        Args:
            thresholds (dict, optional): Dictionary override default thresholds.
                                         Keys like 'soc_low', 'vbus_low',
                                         'ang_vel_high', 'temp_a_low', 'temp_a_high',
                                         'temp_a_critical_low', 'temp_a_critical_high'.
        """
        default_thresholds = {
            'soc_low': 0.15,          # Battery SoC threshold for critical action
            'soc_critical': 0.10,     # Lower threshold for entering Safe Mode
            'vbus_low': 25.0,         # Bus voltage low threshold
            'ang_vel_high': 0.4,      # Angular velocity magnitude (rad/s) high threshold
            'attitude_error_high': np.radians(10.0), # ~10 deg
            'gyro_bias_suspected_error': np.radians(1.0), # If error > ~1 deg, maybe try resetting bias?
            'temp_a_low': 5.0,        # Component A low temp threshold for heater ON
            'temp_a_high': 45.0,       # Component A high temp threshold for heater OFF
            'temp_a_critical_low': -10.0, # Component A critical low temp for recovery
            'temp_a_critical_high': 60.0, # Component A critical high temp for recovery
            # Add thresholds for Temp B, etc. if needed
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
        # Store internal state for mode logic
        self.current_mode = 'Nominal'

    def get_action(self, observation: np.ndarray, current_info: dict) -> int:
        """
        Determine the action based on the current observation and predefined rules.

        Args:
            observation (np.ndarray): The current environment observation (telemetry).
            current_info (dict): Additional info (like fault status, not used by this simple agent).

        Returns:
            int: The discrete action to take (0-8).
        """
        # --- Extract relevant telemetry values --- 
        # Indices for raw_obs (length 15):
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

        # --- Apply Rules (Prioritized) --- 

        # 0. Mode Management - Enter Safe Mode if critical power
        if eps_soc < self.thresholds['soc_critical'] and self.current_mode == 'Nominal':
            print("[Classical FDIR] Rule Triggered: Critical SoC. Action: ENTER_SAFE_MODE")
            self.current_mode = 'Safe' # Update internal tracker
            return self.ACTION_MAP["ENTER_SAFE_MODE"]
        # Try to return to Nominal if conditions improve?
        if eps_soc > (self.thresholds['soc_critical'] + 0.05) and self.current_mode == 'Safe':
             print("[Classical FDIR] Rule Triggered: SoC Recovered. Action: ENTER_NOMINAL_MODE")
             self.current_mode = 'Nominal'
             return self.ACTION_MAP["ENTER_NOMINAL_MODE"]

        # 1. Critical EPS Recovery (Voltage based)
        # If voltage is low despite being in Safe Mode, try full recovery
        if eps_vbus < self.thresholds['vbus_low']:
            print("[Classical FDIR] Rule Triggered: Critical VBus Low. Action: RECOVER_EPS")
            return self.ACTION_MAP["RECOVER_EPS"]

        # 2. Critical ADCS Recovery (Tumbling or Large Error)
        is_tumbling = adcs_ang_vel_mag > self.thresholds['ang_vel_high']
        has_large_error = adcs_att_err_angle > self.thresholds['attitude_error_high']
        if is_tumbling or has_large_error:
            # If error is large but not tumbling, maybe just bias reset first?
            if has_large_error and not is_tumbling and adcs_att_err_angle < np.radians(20.0): # Arbitrary threshold
                 # Check if GyroBias is the *only* potential ADCS issue based on status?
                 adcs_status = subsystem_statuses.get('ADCS', 'Nominal')
                 if adcs_status == 'GyroBias' or adcs_status == 'Nominal': # Try bias reset if bias fault or no specific fault known
                    print("[Classical FDIR] Rule Triggered: Large Attitude Error, attempting Gyro Bias Reset.")
                    return self.ACTION_MAP["RESET_GYRO_BIAS"]
            # Otherwise (tumbling or very large error), perform full recovery
            print("[Classical FDIR] Rule Triggered: High angular velocity or very large attitude error. Action: RECOVER_ADCS")
            return self.ACTION_MAP["RECOVER_ADCS"]
            
        # 3. Moderate Attitude Error - Try Bias Reset
        # If error is moderate, not tumbling, and bias *might* be the cause
        if not is_tumbling and adcs_att_err_angle > self.thresholds['gyro_bias_suspected_error']:
            # Check if GyroBias fault is active OR if no persistent fault is active
            adcs_status = subsystem_statuses.get('ADCS', 'Nominal')
            # Let's be more conservative: only reset bias if specifically suspected or nominal
            if adcs_status == 'GyroBias' or adcs_status == 'Nominal': 
                print("[Classical FDIR] Rule Triggered: Moderate Attitude Error. Action: RESET_GYRO_BIAS")
                return self.ACTION_MAP["RESET_GYRO_BIAS"]

        # 4. Critical TCS Recovery (Out of safety bounds)
        if tcs_temp_a < self.thresholds['temp_a_critical_low'] or tcs_temp_a > self.thresholds['temp_a_critical_high']:
            print("[Classical FDIR] Rule Triggered: Critical TCS Temp A. Action: RECOVER_TCS")
            return self.ACTION_MAP["RECOVER_TCS"]

        # 5. TCS Heater Control (Nominal Temperature Regulation)
        # Only control heater if TCS fault doesn't prevent it (e.g., not StuckOn/Off)
        tcs_status = subsystem_statuses.get('TCS', 'Nominal')
        can_control_heater = 'HeaterStuck' not in tcs_status
        if can_control_heater:
            if tcs_temp_a < self.thresholds['temp_a_low'] and tcs_heater_status < 0.5:
                print("[Classical FDIR] Rule Triggered: Low TCS Temp A. Action: HEATER_ON")
                return self.ACTION_MAP["HEATER_ON"]
            if tcs_temp_a > self.thresholds['temp_a_high'] and tcs_heater_status > 0.5:
                print("[Classical FDIR] Rule Triggered: High TCS Temp A. Action: HEATER_OFF")
                return self.ACTION_MAP["HEATER_OFF"]

        # --- Default Action --- 
        return self.ACTION_MAP["NO_OP"] 
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SpacecraftEnv(gym.Env):
    """
    A simplified version of the spacecraft environment for testing the hybrid agent comparison.
    This environment simulates basic spacecraft dynamics and allows for fault injection.
    """
    
    def __init__(self, fault_probability=0.02, fault_sample_mode="random"):
        super().__init__()
        
        # Environment configuration
        self.fault_probability = fault_probability
        self.fault_sample_mode = fault_sample_mode
        
        # Action space: 9 discrete actions
        # 0: No-op, 1-3: Recovery actions, 4-6: Subsystem controls, 7-8: Mode changes
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 12-dimensional vector of telemetry values
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Internal state
        self.active_fault = None
        self.state = np.zeros(12, dtype=np.float32)
        self.step_count = 0
        
        # Action mapping for compatibility with hybrid agent
        self.ACTION_MAP = {
            "NO_OP": 0,
            "RECOVER_EPS": 1,
            "RECOVER_ADCS": 2,
            "RECOVER_TCS": 3,
            "HEATER_ON": 4,
            "HEATER_OFF": 5,
            "RESET_GYRO": 6,
            "ENTER_SAFE_MODE": 7,
            "ENTER_NOMINAL": 8
        }
    
    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state to nominal values with small random variations
        self.state = 0.5 + np.random.normal(0, 0.05, size=12)
        self.state = np.clip(self.state, 0.0, 1.0)
        
        # No active fault at start
        self.active_fault = None
        self.step_count = 0
        
        # Return observation and info
        return self.state.copy(), self._get_info()
    
    def step(self, action):
        """Execute one time step of the environment."""
        self.step_count += 1
        
        # Apply action effects
        if action in [1, 2, 3]:  # Recovery actions
            if self.active_fault and action - 1 == self.active_fault["subsystem_id"]:
                self.active_fault = None
        
        # Simple state evolution
        self.state += np.random.normal(0, 0.03, size=12)
        
        # Randomly inject fault with probability
        fault_injected = False
        if self.active_fault is None and np.random.random() < self.fault_probability:
            fault_subsystem = np.random.randint(0, 3)  # 0=EPS, 1=ADCS, 2=TCS
            
            # Inject fault effect into state
            if fault_subsystem == 0:  # EPS fault
                self.state[0] -= 0.3  # Lower battery state
            elif fault_subsystem == 1:  # ADCS fault
                self.state[4] += 0.4  # Increase attitude error
            else:  # TCS fault
                self.state[8] += 0.4  # Increase temperature
            
            self.active_fault = {
                "subsystem_id": fault_subsystem,
                "subsystem": ["EPS", "ADCS", "TCS"][fault_subsystem],
                "type": f"{['EPS', 'ADCS', 'TCS'][fault_subsystem]}Fault"
            }
            fault_injected = True
        
        # Update state based on active fault
        if self.active_fault:
            fault_subsystem = self.active_fault["subsystem_id"]
            if fault_subsystem == 0:  # EPS fault
                self.state[0] -= 0.05  # Continuing battery drain
            elif fault_subsystem == 1:  # ADCS fault
                self.state[4] += 0.05  # Worsening attitude error
            else:  # TCS fault
                self.state[8] += 0.05  # Rising temperature
        
        # Calculate reward (negative penalties)
        reward = -np.sum(np.abs(self.state - 0.5)) * 0.5
        
        # Check for terminal conditions
        done = False
        if np.any(self.state < 0.0) or np.any(self.state > 1.0):
            done = True
            reward -= 10.0  # Large penalty for system failure
        
        # Clip state to bounds
        self.state = np.clip(self.state, 0.0, 1.0)
        
        # Get info dict
        info = self._get_info()
        if fault_injected:
            info["fault_injected"] = True
            info["fault_type"] = self.active_fault["type"]
            info["fault_subsystem"] = self.active_fault["subsystem"]
        
        return self.state.copy(), reward, done, False, info
    
    def _get_info(self):
        """Return information about the current state."""
        subsystem_statuses = {
            "EPS": "Fault" if self.active_fault and self.active_fault["subsystem_id"] == 0 else "Nominal",
            "ADCS": "Fault" if self.active_fault and self.active_fault["subsystem_id"] == 1 else "Nominal", 
            "TCS": "Fault" if self.active_fault and self.active_fault["subsystem_id"] == 2 else "Nominal"
        }
        
        return {
            "subsystem_statuses": subsystem_statuses,
            "step": self.step_count
        }
    
    def has_active_fault(self):
        """Return whether there is an active fault in the system."""
        return self.active_fault is not None
    
    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        np.random.seed(seed)
        return [seed] 
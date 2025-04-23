import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random # For fault injection probability maybe

from .subsystems import ElectricalPowerSystem, AttitudeControlSystem, ThermalControlSystem, OrbitModel # Import OrbitModel
from .faults import FaultInjector, POSSIBLE_FAULTS

# Helper function for normalization
def normalize(value, low, high):
    """Normalize value to [-1, 1] range."""
    if high == low: # Avoid division by zero
        return 0.0 if value == low else np.sign(value - low)
    return 2 * (value - low) / (high - low) - 1

def denormalize(norm_value, low, high):
    """Denormalize value from [-1, 1] range."""
    return ((norm_value + 1) / 2) * (high - low) + low

class SpacecraftEnv(gym.Env):
    """Custom Environment for Spacecraft FDIR simulation following Gymnasium API."""
    metadata = {'render_modes': ['human', 'ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None, fault_probability=0.01, max_steps=1000, normalize_obs=True):
        super().__init__()

        self.max_steps = max_steps
        self.current_step = 0
        self.mode = 'Nominal'
        self.normalize_obs = normalize_obs

        # --- Initialize Orbit Model ---
        self.orbit_model = OrbitModel()

        # --- Initialize Subsystems ---
        self.eps = ElectricalPowerSystem()
        self.adcs = AttitudeControlSystem()
        self.tcs = ThermalControlSystem()
        self.subsystems = {
            "EPS": self.eps,
            "ADCS": self.adcs,
            "TCS": self.tcs
        }
        self.fault_injector = FaultInjector(self.subsystems, fault_probability)

        # --- Define Action Space ---
        # 0: No-op
        # 1: Recover EPS (Full Reset)
        # 2: Recover ADCS (Full Reset)
        # 3: Recover TCS (Full Reset)
        # 4: Command TCS Heater ON
        # 5: Command TCS Heater OFF
        # 6: Reset ADCS Gyro Bias Estimate 
        # 7: Enter Safe Mode
        # 8: Enter Nominal Mode
        self.action_space = spaces.Discrete(9)

        # --- Define Observation Space --- 
        # Concatenate telemetry from all subsystems
        # EPS: [SoC(0-1), P_potential(W), V_bus(V), ChargeRate(W)] - 4
        # ADCS: [q1,q2,q3,q0 (-1 to 1), wx,wy,wz (rad/s), att_err(rad)] - 8 (+1)
        # TCS: [TempA(C), TempB(C), HeaterStatus(0/1)] - 3
        # Total size: 4 + 8 + 3 = 15

        # Store raw bounds for potential normalization/denormalization
        self.raw_obs_bounds_low = np.array([
            0.0, 0.0, 24.0, -100.0,       # EPS Low bounds (SoC, P_pot, V_bus, ChargeRate)
           -1.0,-1.0,-1.0,-1.0,          # ADCS Quaternions
           -np.pi, -np.pi, -np.pi, 0.0,   # ADCS AngVel(rad/s), AttErrAngle(rad, >=0)
           -50.0, -50.0, 0.0              # TCS Temps (C), HeaterStatus
        ], dtype=np.float32)

        self.raw_obs_bounds_high = np.array([
            1.0, 150.0, 32.0, 150.0,       # EPS High bounds (SoC, P_pot, V_bus, ChargeRate)
            1.0, 1.0, 1.0, 1.0,           # ADCS Quaternions
            np.pi, np.pi, np.pi, np.pi,   # ADCS AngVel(rad/s), AttErrAngle(rad, <=pi)
            100.0, 100.0, 1.0             # TCS Temps (C), HeaterStatus
        ], dtype=np.float32)
        
        if self.normalize_obs:
            # Normalized space is typically [-1, 1] for each element
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(15,), dtype=np.float32
            )
        else:
             # Use raw bounds if not normalizing
            self.observation_space = spaces.Box(
                low=self.raw_obs_bounds_low, high=self.raw_obs_bounds_high, dtype=np.float32
            )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self) -> np.ndarray:
        """Helper function to gather observations from all subsystems."""
        eps_telemetry = self.eps.get_telemetry()       # Shape (4,)
        adcs_telemetry = self.adcs.get_telemetry()      # Shape (8,)
        tcs_telemetry = self.tcs.get_telemetry()       # Shape (3,)
        raw_obs = np.concatenate((eps_telemetry, adcs_telemetry, tcs_telemetry)).astype(np.float32)
        
        # Clip raw obs to bounds before normalization (important!)
        raw_obs_clipped = np.clip(raw_obs, self.raw_obs_bounds_low, self.raw_obs_bounds_high)

        if self.normalize_obs:
            # Normalize each element using its specific bounds
            normalized_obs = np.array([
                normalize(raw_obs_clipped[i], self.raw_obs_bounds_low[i], self.raw_obs_bounds_high[i]) 
                for i in range(len(raw_obs_clipped))
            ], dtype=np.float32)
            return normalized_obs
        else:
            return raw_obs_clipped # Return clipped raw obs if not normalizing

    def _get_info(self) -> dict:
        """Helper function to gather additional info (e.g., fault status)."""
        active_faults_summary = self.fault_injector.get_active_faults_summary()
        # Get current subsystem status (might reflect intermittent)
        subsystem_statuses = {
            name: sub.fault_status 
            if not sub.is_intermittent_fault_active 
            else sub.active_intermittent_fault_type 
            for name, sub in self.subsystems.items()
        }
        info_dict = {
            "subsystem_statuses": subsystem_statuses, # Reflects current apparent status
            "active_faults_persistent": active_faults_summary['persistent'],
            "active_faults_intermittent": active_faults_summary['intermittent_active'],
            "is_sunlit": self.orbit_model.is_sunlit,
            "mode": self.mode
        }
        if self.normalize_obs:
             eps_telemetry = self.eps.get_telemetry()
             adcs_telemetry = self.adcs.get_telemetry()
             tcs_telemetry = self.tcs.get_telemetry()
             raw_obs = np.concatenate((eps_telemetry, adcs_telemetry, tcs_telemetry)).astype(np.float32)
             info_dict["raw_observation"] = np.round(raw_obs, 3).tolist()
        return info_dict

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0
        self.mode = 'Nominal' # Reset mode

        # Reset all subsystems to nominal
        for subsystem in self.subsystems.values():
            subsystem.recover() # Use recover to reset to nominal state

        # Reset fault injector (clears its lists and subsystem flags)
        self.fault_injector.reset()

        # Get initial statuses (should be nominal)
        self.previous_subsystem_statuses = {
             name: sub.fault_status 
             if not sub.is_intermittent_fault_active 
             else sub.active_intermittent_fault_type 
             for name, sub in self.subsystems.items()
         }

        # Reset orbit model
        self.orbit_model.reset()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """Execute one time step within the environment."""
        self.current_step += 1
        terminated = False
        truncated = False
        # Get status before action/update
        current_subsystem_statuses = {
             name: sub.fault_status 
             if not sub.is_intermittent_fault_active 
             else sub.active_intermittent_fault_type 
             for name, sub in self.subsystems.items()
         }

        # --- Determine Mode (Now only changed by explicit actions) --- 
        # Mode persists unless changed by action 7 or 8
        
        # --- Apply Action --- 
        action_penalty = 0.0
        tcs_action = None # Default unless action 4 or 5 is chosen
        
        if action == 1: # Recover EPS
            if current_subsystem_statuses["EPS"] == 'Nominal': action_penalty -= 0.5
            self.eps.recover()
        elif action == 2: # Recover ADCS
            if current_subsystem_statuses["ADCS"] == 'Nominal': action_penalty -= 0.5
            self.adcs.recover()
        elif action == 3: # Recover TCS
            if current_subsystem_statuses["TCS"] == 'Nominal': action_penalty -= 0.5
            self.tcs.recover()
        elif action == 4: # Command Heater ON
            tcs_action = {'heater_command': 1.0}
            # Penalize if heater already on or stuck?
            # if self.tcs.state[2] > 0.5: action_penalty -= 0.1
        elif action == 5: # Command Heater OFF
            tcs_action = {'heater_command': 0.0}
            # Penalize if heater already off or stuck?
            # if self.tcs.state[2] < 0.5: action_penalty -= 0.1
        elif action == 6: # Reset ADCS Gyro Bias Estimate
            self.adcs.reset_gyro_bias()
            # Penalize if no gyro bias fault suspected? Maybe not, could be preventative.
        elif action == 7: # Enter Safe Mode
            if self.mode == 'Safe': action_penalty -= 0.1 # Penalize redundant mode change
            self.mode = 'Safe'
            print("Info: Entering Safe Mode.")
        elif action == 8: # Enter Nominal Mode
            if self.mode == 'Nominal': action_penalty -= 0.1 # Penalize redundant mode change
            self.mode = 'Nominal'
            print("Info: Entering Nominal Mode.")
        # Action 0 is No-op, handled by default

        # --- Update Orbit --- 
        dt = 1.0 # Simulation time step in seconds
        self.orbit_model.update(dt)
        spacecraft_state_info = {'is_sunlit': self.orbit_model.is_sunlit, 'mode': self.mode}

        # --- Update Subsystems --- 
        self.eps.update(dt, spacecraft_state=spacecraft_state_info)
        self.adcs.update(dt, spacecraft_state=spacecraft_state_info)
        self.tcs.update(dt, spacecraft_state=spacecraft_state_info, action=tcs_action)

        # --- Update Faults (Persistent Intro + Intermittent State Changes) --- 
        self.fault_injector.step() # IMPORTANT: Let fault states update *after* subsystems update
                                # but *before* getting the final observation for the agent.

        # --- Get Observation & Info (Reflects post-fault-update state) --- 
        observation = self._get_obs() 
        info = self._get_info()
        # Get the apparent status *after* fault updates for reward calculation
        new_subsystem_statuses = info["subsystem_statuses"] 

        # --- Calculate Reward --- 
        reward = 0.0
        reward += action_penalty
        raw_obs = info.get("raw_observation", observation) if self.normalize_obs else observation
        # Indices for raw_obs (length 15):
        eps_soc             = raw_obs[0]
        # eps_p_potential     = raw_obs[1] # Not used in reward currently
        # eps_vbus            = raw_obs[2] # Not used in reward currently
        # eps_charge_rate     = raw_obs[3] # Not used in reward currently
        # adcs_q              = raw_obs[4:8] # Not used in reward currently
        adcs_w_xyz          = raw_obs[8:11]
        adcs_att_err_angle  = raw_obs[11]
        tcs_temp_a          = raw_obs[12]
        # tcs_temp_b          = raw_obs[13] # Not used in reward currently
        tcs_heater_status   = raw_obs[14]
        
        adcs_ang_vel_mag = np.linalg.norm(adcs_w_xyz)
        
        all_nominal = all(status == 'Nominal' for status in new_subsystem_statuses.values())
        
        # --- Base Reward/Penalty for Overall State ---
        if all_nominal and self.mode == 'Nominal':
            # Reward for being fully nominal
            reward += 0.1 
            # Add small reward for low attitude error even when nominal
            reward += 0.1 * (1.0 - np.clip(adcs_att_err_angle / np.radians(5.0), 0, 1)) # Max reward if error < 5 deg
        else:
             # Small base penalty for not being fully nominal
             reward -= 0.1 

        # --- Penalties/Rewards for Specific Subsystem States --- 
        # EPS
        if new_subsystem_statuses['EPS'] != 'Nominal':
            reward -= 1.0 # Penalty for EPS fault active
            if eps_soc < 0.1: # Critical low battery penalty
                reward -= (10.0 if not info['is_sunlit'] else 5.0)
        elif self.previous_subsystem_statuses['EPS'] != 'Nominal': 
            reward += 2.0 # Reward successful EPS recovery

        # ADCS
        if new_subsystem_statuses['ADCS'] != 'Nominal':
            reward -= 1.0 # Penalty for ADCS fault active
        elif self.previous_subsystem_statuses['ADCS'] != 'Nominal':
            reward += 2.0 # Reward successful ADCS recovery
        # Penalties for poor ADCS performance (even if subsystem status is 'Nominal')
        if adcs_ang_vel_mag > 0.5: # Tumbling penalty
             reward -= 5.0 
        if adcs_att_err_angle > np.radians(10.0): # Penalty if error > ~10 deg
             reward -= 5.0 * (adcs_att_err_angle / np.pi) # Scale penalty by error magnitude

        # TCS
        tcs_status = new_subsystem_statuses['TCS']
        if tcs_status != 'Nominal':
            reward -= 1.0 # Penalty for TCS fault active
            if tcs_temp_a < 0.0 or tcs_temp_a > 50.0: # Temp violation penalty
                reward -= 3.0
            # Penalties for incorrect heater state during faults
            if tcs_status == 'HeaterStuckOn' and tcs_heater_status < 0.5: reward -= 2.0
            elif tcs_status == 'HeaterStuckOff' and tcs_heater_status > 0.5: reward -= 2.0
        elif self.previous_subsystem_statuses['TCS'] != 'Nominal':
            reward += 2.0 # Reward successful TCS recovery

        # --- Penalty for being in Safe mode --- 
        if self.mode == 'Safe':
             reward -= 0.2

        # --- Check Termination/Truncation Conditions --- 
        if self.current_step >= self.max_steps:
            truncated = True
            
        # Use raw values for termination checks
        is_degraded = self.eps.fault_status == 'SolarPanelDegradation'
        if eps_soc < 0.05 and not info['is_sunlit'] and is_degraded:
             print("Termination Condition: Critical Power Loss (Eclipse, Degraded)")
             terminated = True; reward -= 50
        elif eps_soc < 0.01:
             print("Termination Condition: Critical Power Loss (Absolute Minimum SoC)")
             terminated = True; reward -= 50
        if adcs_att_err_angle > np.radians(45):
            print("Termination Condition: Excessive Attitude Error")
            terminated = True; reward -= 50

        # Update previous status for the next step's calculation
        self.previous_subsystem_statuses = new_subsystem_statuses
        if self.render_mode == "human":
            self._render_frame()
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment state."""
        if self.render_mode == "ansi":
            # Simple text-based rendering
            obs = self._get_obs()
            info = self._get_info()
            fault_str = ", ".join([f"{name}:{status}" for name, status in info['subsystem_statuses'].items() if status != 'Nominal'])
            if not fault_str: fault_str = "None"
            return f"Step: {self.current_step}, Faults: [{fault_str}], Obs: {np.round(obs, 2)}"
        elif self.render_mode == "human":
            # Placeholder for a more visual rendering (e.g., using Pygame or Matplotlib)
            # For now, just print the ANSI representation
            print(self.render(render_mode='ansi'))

    def _render_frame(self):
        # Internal helper for human rendering (currently prints text)
        print(self.render(render_mode='ansi'))

    def close(self):
        """Perform any necessary cleanup."""
        # Close any open resources (e.g., rendering windows)
        pass 
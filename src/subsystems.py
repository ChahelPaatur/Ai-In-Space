import numpy as np

# --- Orbital Simulation Constants (Simplified) ---
# Assuming a generic Low Earth Orbit (LEO) for example purposes
ORBIT_PERIOD_SECONDS = 90 * 60 # 90 minutes
SUNLIGHT_FRACTION = 0.6 # Approx fraction of orbit in sunlight
SUNLIGHT_DURATION = ORBIT_PERIOD_SECONDS * SUNLIGHT_FRACTION
ECLIPSE_DURATION = ORBIT_PERIOD_SECONDS * (1.0 - SUNLIGHT_FRACTION)

class OrbitModel:
    """Very simple model to track simulation time and determine sun/eclipse periods."""
    def __init__(self):
        self.simulation_time_seconds = 0.0
        self.is_sunlit = True # Start in sunlight

    def update(self, dt: float):
        self.simulation_time_seconds += dt
        # Calculate position within the current orbit cycle
        time_in_cycle = self.simulation_time_seconds % ORBIT_PERIOD_SECONDS
        # Determine if sunlit based on simple fractional duration
        self.is_sunlit = time_in_cycle < SUNLIGHT_DURATION

    def reset(self):
        self.simulation_time_seconds = 0.0
        self.is_sunlit = True

class BaseSubsystem:
    """Base class for all spacecraft subsystems."""
    def __init__(self, name: str, state_size: int):
        self.name = name
        self.state = np.zeros(state_size)
        self.nominal_state = np.zeros(state_size)
        self.fault_status = 'Nominal' # Tracks the primary *persistent* fault type
        self._internal_fault_params = {} 
        # Intermittent fault state
        self.is_intermittent_fault_active = False
        self.active_intermittent_fault_type = None
        self.active_intermittent_fault_params = {}

    def update(self, dt: float, spacecraft_state: dict, action: dict | None = None):
        """Update the subsystem state over a time step dt.

        Args:
            dt (float): Time step in seconds.
            spacecraft_state (dict): Dictionary containing overall spacecraft state info 
                                     (e.g., {'is_sunlit': True, 'mode': 'Nominal'}).
            action (dict | None): Specific actions for the subsystem.
        """
        # Basic placeholder: state doesn't change unless overridden
        pass

    def get_telemetry(self) -> np.ndarray:
        """Return the current telemetry values for this subsystem."""
        # Base implementation: return state. Subclasses should add noise/sensor models.
        return self.state.copy()

    def apply_fault(self, fault_info: dict):
        """Apply a *persistent* fault to the subsystem."""
        # This method now only handles persistent faults
        is_intermittent = fault_info.get('intermittent', False)
        if is_intermittent:
             print(f"Warning: apply_fault called with intermittent fault info for {self.name}. Ignoring.")
             return 
             
        fault_type = fault_info.get('type', 'Unknown Persistent Fault')
        params = fault_info.get('params', {})
        self.fault_status = fault_type # Set persistent fault status
        self._internal_fault_params = params
        print(f"Applied Persistent Fault: {fault_type} to {self.name} with params: {params}")
        self._apply_fault_effect() # Apply immediate effects if any
        # Clear any active intermittent state when a persistent fault is applied
        self.deactivate_intermittent_fault(None)

    def activate_intermittent_fault(self, fault_type: str, params: dict):
        """Activates an intermittent fault effect."""
        self.is_intermittent_fault_active = True
        self.active_intermittent_fault_type = fault_type
        self.active_intermittent_fault_params = params
        # Apply immediate effect if necessary (optional, depends on fault)
        # self._apply_intermittent_fault_effect() 

    def deactivate_intermittent_fault(self, fault_type: str | None):
        """Deactivates a specific or all intermittent fault effects."""
        # If fault_type is None, deactivate any active intermittent fault
        if fault_type is None or self.active_intermittent_fault_type == fault_type:
            self.is_intermittent_fault_active = False
            self.active_intermittent_fault_type = None
            self.active_intermittent_fault_params = {}
            # Reverse immediate effect if necessary (optional)
            # self._reverse_intermittent_fault_effect()

    def _apply_fault_effect(self):
         """Internal method for subclasses to implement immediate persistent fault effects."""
         pass

    def clear_fault_state(self):
        """Clears persistent fault state *without* resetting the whole subsystem state."""
        print(f"Info: Clearing persistent fault state for {self.name}. Was: {self.fault_status}")
        self.fault_status = 'Nominal'
        self._internal_fault_params = {}
        # Reset any internal variables affected by persistent faults (e.g., noise levels)
        self._reset_persistent_fault_internals()

    def _reset_persistent_fault_internals(self):
         """Subclasses implement resetting internal vars affected by persistent faults."""
         pass
         
    def recover(self):
        """Full recovery: reset state, clear persistent & intermittent faults."""
        print(f"Info: Recovery action triggered for {self.name}. Resetting state & faults.")
        self.state = self.nominal_state.copy()
        self.clear_fault_state() # Clear persistent fault info
        self.deactivate_intermittent_fault(None) # Clear intermittent fault info
        # Reset any other internal state (like charge rate)
        self._reset_internal_state()
        
    def _reset_internal_state(self):
        """Subclasses implement resetting other internal state during full recovery."""
        pass

class ElectricalPowerSystem(BaseSubsystem):
    """Simplified Electrical Power System (EPS) with variable load and voltage dynamics."""
    def __init__(self):
        # State: [Battery SoC (0-1), Solar Array Power Gen Potential(W), Bus Voltage (V)]
        state_size = 3
        super().__init__("EPS", state_size)
        self.nominal_state = np.array([0.8, 100.0, 28.0])
        self.state = self.nominal_state.copy()
        # Define power loads for different modes (W)
        self._load_nominal = 50.0
        self._load_safe_mode = 25.0
        self._current_load = self._load_nominal
        self._battery_capacity_wh = 1000.0
        self._actual_battery_capacity_wh = self._battery_capacity_wh
        self.charge_discharge_rate_w = 0.0 # Add charge rate state

    def update(self, dt: float, spacecraft_state: dict, action: dict | None = None):
        dt_hours = dt / 3600.0
        is_sunlit = spacecraft_state.get('is_sunlit', True)
        mode = spacecraft_state.get('mode', 'Nominal')

        # Update load based on mode
        self._current_load = self._load_safe_mode if mode == 'Safe' else self._load_nominal

        # Apply persistent fault effects during update
        solar_power_potential = self.state[1]
        if self.fault_status == 'SolarPanelDegradation':
            degradation = self._internal_fault_params.get('degradation_factor', 1.0)
            solar_power_potential *= degradation

        # Generate power only if sunlit
        generated_power = solar_power_potential if is_sunlit else 0.0

        net_power = generated_power - self._current_load
        self.charge_discharge_rate_w = net_power # Store for telemetry

        # Update battery state of charge (SoC)
        energy_change_wh = net_power * dt_hours
        soc_change = energy_change_wh / self._actual_battery_capacity_wh if self._actual_battery_capacity_wh > 0 else 0
        current_soc = self.state[0]
        new_soc = np.clip(current_soc + soc_change, 0.0, 1.0)
        self.state[0] = new_soc

        # --- Refined Bus Voltage Logic --- 
        # Base voltage depends on SoC (simple linear model)
        base_voltage = 26.0 + 4.0 * new_soc # Example: 26V at 0% SoC, 30V at 100% SoC

        # Adjust voltage based on load/charge (Ohm's law approximation V = V_oc - I*R_internal)
        # Higher discharge current -> lower voltage, Higher charge current -> higher voltage
        # Simple adjustment based on net_power (needs tuning)
        internal_resistance_effect = 0.01 # Fictional internal resistance factor
        voltage_adjustment = -net_power * internal_resistance_effect
        bus_voltage = base_voltage + voltage_adjustment
        # Clamp voltage to reasonable bounds
        bus_voltage = np.clip(bus_voltage, 24.0, 32.0)

        # Apply specific fault conditions
        if self.fault_status == 'BatteryCellFailure' and self._actual_battery_capacity_wh < 1:
            bus_voltage = 24.0 # Override if battery effectively dead
        elif self.fault_status == 'PowerShortCircuit': # Example new fault
            bus_voltage = 10.0 # Drastic voltage drop
            self._current_load *= 3 # Simulate high current draw

        self.state[2] = bus_voltage

    def get_telemetry(self) -> np.ndarray:
        """Return telemetry including charge rate."""
        base_telemetry = super().get_telemetry()
        # Return [SoC, P_potential, V_bus, Charge/Discharge Rate W]
        return np.append(base_telemetry, self.charge_discharge_rate_w)

    def _apply_fault_effect(self):
        """Apply immediate effects of an EPS fault."""
        if self.fault_status == 'SolarPanelDegradation':
            # The effect is applied during update by reducing potential
            print(f"{self.name}: Solar panel potential will be degraded.")
        elif self.fault_status == 'BatteryCellFailure':
            reduction = self._internal_fault_params.get('capacity_reduction', 1.0)
            self._actual_battery_capacity_wh = self._battery_capacity_wh * (1.0 - reduction)
            current_energy_wh = self.state[0] * self._actual_battery_capacity_wh
            self.state[0] = current_energy_wh / self._actual_battery_capacity_wh if self._actual_battery_capacity_wh > 0 else 0
            self.state[0] = np.clip(self.state[0], 0.0, 1.0)
            print(f"{self.name}: Battery capacity reduced to {self._actual_battery_capacity_wh:.2f} Wh.")
        elif self.fault_status == 'PowerShortCircuit':
             print(f"{self.name}: Power system short circuit applied.")
             # Immediate effect on voltage/load is handled in update()
        # Add other fault effects here

    def _reset_persistent_fault_internals(self):
        self._actual_battery_capacity_wh = self._battery_capacity_wh
        # Reset load effect from potential short circuit? Assume recover handles this.
        
    def _reset_internal_state(self):
        self.charge_discharge_rate_w = 0.0
        self._current_load = self._load_nominal

class AttitudeControlSystem(BaseSubsystem):
    """Simplified ADCS with basic kinetics and sensor noise."""
    def __init__(self):
        # State: [Quaternion (q1,q2,q3,q0), Angular Velocity (wx,wy,wz)] - 7 states
        state_size = 7
        super().__init__("ADCS", state_size)
        self.nominal_state = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) # Inertial pointing, zero rates
        self.state = self.nominal_state.copy()
        self._gyro_bias = np.zeros(3)
        self._reaction_wheel_friction_factor = 1.0

        # --- Add Inertia Tensor (Example: Diagonal for simplicity) ---
        # Represents resistance to rotational changes (kg*m^2)
        # These values are placeholders and depend heavily on spacecraft geometry/mass distribution
        self.inertia_tensor = np.diag([10.0, 10.0, 5.0]) 
        self.inv_inertia_tensor = np.linalg.inv(self.inertia_tensor)

        # --- Add Sensor Noise Characteristics (Example) ---
        self._base_gyro_noise_std_dev = 0.001 # Store the base noise level
        self.current_gyro_noise_std_dev = self._base_gyro_noise_std_dev # Current noise level used
        
        # --- Target Attitude (for error calculation) ---
        # Could be dynamic (e.g., pointing target) or fixed (e.g., inertial hold)
        self.target_quaternion = np.array([0.0, 0.0, 0.0, 1.0]) # Target: zero rotation

    def update(self, dt: float, spacecraft_state: dict, action: dict | None = None):
        q = self.state[0:4] # Current quaternion [q1, q2, q3, q0]
        w = self.state[4:7] # Current angular velocity [wx, wy, wz] (rad/s)

        # --- Calculate External/Internal Torques --- 
        # Placeholder for more complex models (gravity gradient, solar pressure, etc.)
        external_torque = np.zeros(3)

        # --- Apply Control Torques (Simplified Example) ---
        # Action could specify desired torque (not implemented in action space yet)
        # Or, a simple stabilization torque could be applied if not nominal
        control_torque = np.zeros(3)
        # Example: Simple proportional damping if tumbling (needs tuning)
        if np.linalg.norm(w) > 0.1: 
            control_torque = -0.1 * w # Apply torque opposing angular velocity

        # --- Apply Fault Effects on Dynamics ---
        friction_torque = np.zeros(3)
        if self.fault_status == 'ReactionWheelFriction':
            # Simulate increased friction as a torque opposing motion
            friction_factor = self._internal_fault_params.get('friction_increase', 1.0)
            friction_torque = -0.05 * friction_factor * w # Simple model
            # Also apply decay as before? Or rely solely on torque?
            # Let's use torque primarily for friction now
            # decay = 0.99 ** (friction_factor * dt) 
            # w *= decay 

        # --- Euler's Equation of Motion (Rotational Dynamics) --- 
        # dw/dt = I^-1 * (Total Torque - w x (I*w))
        # Total Torque = Control + External + Friction + Other Perturbations
        total_torque = control_torque + external_torque + friction_torque
        # Coriolis term: w x (I*w)
        inertia_times_w = np.dot(self.inertia_tensor, w)
        coriolis_torque = np.cross(w, inertia_times_w)
        
        # Calculate angular acceleration (alpha = dw/dt)
        angular_acceleration = np.dot(self.inv_inertia_tensor, (total_torque - coriolis_torque))

        # Integrate angular velocity using Euler method
        w_new = w + angular_acceleration * dt

        # --- Quaternion Kinematics --- 
        # Use the *average* angular velocity over the step for better accuracy?
        w_avg = 0.5 * (w + w_new) 
        wx, wy, wz = w_avg
        omega_matrix = np.array([
            [ 0.,  wz, -wy,  wx],
            [-wz,  0.,  wx,  wy],
            [ wy, -wx,  0.,  wz],
            [-wx, -wy, -wz,  0.]
        ])
        q_dot = 0.5 * np.dot(omega_matrix, q)
        q_new = q + q_dot * dt
        norm = np.linalg.norm(q_new)
        if norm > 1e-6: q_new /= norm
        else: q_new = np.array([0., 0., 0., 1.])

        # Update the state vector
        self.state[0:4] = q_new
        self.state[4:7] = w_new # Use the updated angular velocity

    def get_telemetry(self) -> np.ndarray:
        """Return telemetry including sensor noise (potentially increased) and attitude error."""
        # --- Simulate Sensor Readings --- 
        true_angular_velocity = self.state[4:7]
        # Use the current noise level for simulation
        gyro_noise = np.random.normal(0, self.current_gyro_noise_std_dev, size=3)
        measured_angular_velocity = true_angular_velocity + self._gyro_bias + gyro_noise
        measured_quaternion = self.state[0:4]
        
        # --- Calculate Attitude Error --- 
        q_measured_conj = measured_quaternion * np.array([-1, -1, -1, 1])
        t1,t2,t3,t0 = self.target_quaternion
        c1,c2,c3,c0 = q_measured_conj
        q_error = np.array([
            t0*c1 + t1*c0 + t2*c3 - t3*c2,
            t0*c2 - t1*c3 + t2*c0 + t3*c1,
            t0*c3 + t1*c2 - t2*c1 + t3*c0,
            t0*c0 - t1*c1 - t2*c2 - t3*c3
        ])
        attitude_error_angle = 2 * np.arccos(np.clip(q_error[3], -1.0, 1.0))
        
        # --- Assemble Telemetry Vector --- 
        adcs_telemetry = np.concatenate((measured_quaternion, measured_angular_velocity, [attitude_error_angle]))
        return adcs_telemetry

    def _apply_fault_effect(self):
        """Apply immediate effects of an ADCS fault."""
        if self.fault_status == 'GyroBias':
            bias_value = self._internal_fault_params.get('bias_value', 0.0)
            axis = np.random.randint(3)
            self._gyro_bias = np.zeros(3)
            self._gyro_bias[axis] = bias_value
            print(f"{self.name}: Gyro bias applied to axis {axis}.")
        elif self.fault_status == 'ReactionWheelFriction':
            self._reaction_wheel_friction_factor = self._internal_fault_params.get('friction_increase', 1.0)
            print(f"{self.name}: Reaction wheel friction increased (effect applied via torque).")
        elif self.fault_status == 'GyroNoiseIncrease':
            noise_factor = self._internal_fault_params.get('noise_factor', 1.0)
            self.current_gyro_noise_std_dev = self._base_gyro_noise_std_dev * noise_factor
            print(f"{self.name}: Gyro noise increased to {self.current_gyro_noise_std_dev:.4f} rad/s.")
        # Add other fault effects

    def _reset_persistent_fault_internals(self):
        self._gyro_bias = np.zeros(3)
        self._reaction_wheel_friction_factor = 1.0
        self.current_gyro_noise_std_dev = self._base_gyro_noise_std_dev
        
    def _reset_internal_state(self):
        pass # No other internal state specific to ADCS to reset here

    def reset_gyro_bias(self):
        """Resets the internal gyro bias estimate to zero."""
        print(f"Info: Resetting gyro bias for {self.name}.")
        self._gyro_bias = np.zeros(3)
        # Note: This doesn't fix the *cause* if the fault type is still active,
        # but represents an action an FDIR system might take (e.g., based on star tracker data).
        # If the fault is persistent ('GyroBias' status), it might get reapplied by _apply_fault_effect
        # if called again, or the underlying bias source might remain.
        # If the fault was intermittent or transient, this resets the state.
        # Consider if this should also clear the 'GyroBias' fault_status if active?
        # if self.fault_status == 'GyroBias':
        #     self.clear_fault_state() # Optionally clear the persistent fault state too

class ThermalControlSystem(BaseSubsystem):
    """Refined Thermal Control System (TCS) with capacitance and conductance."""
    def __init__(self):
        # State: [ComponentATemp (C), ComponentBTemp (C), HeaterStatus (0=OFF, 1=ON)]
        state_size = 3
        super().__init__("TCS", state_size)
        self.nominal_state = np.array([20.0, 25.0, 0.0]) # Example nominal temperatures, heater off
        self.state = self.nominal_state.copy()
        
        # --- Physical Parameters (Example Values - Need Tuning!) ---
        self.heater_power_w = 10.0 # Heater power (Watts)
        # Thermal Capacitance (Joules / Kelvin)
        self.capacitance_a_J_per_K = 500.0 
        self.capacitance_b_J_per_K = 800.0
        # Conductance between A and B (Watts / Kelvin)
        self.conductance_a_b_W_per_K = 0.5 
        # Linearized Heat Exchange Coefficient with Environment (Watts / Kelvin)
        self.heat_coeff_a_env_W_per_K = 0.1
        self.heat_coeff_b_env_W_per_K = 0.2
        # Internal Heat Generation (Watts) - e.g., from electronics
        self.internal_heat_a_W = 2.0
        self.internal_heat_b_W = 1.0
        # Environment Temperature (Kelvin for calculations, but state is Celsius)
        # Using simplified Celsius ambient temp for linear model consistency
        self._ambient_temp_eclipse_C = -20.0 
        self._ambient_temp_sunlit_C = 0.0 # Effective temp slightly warmer in sun
        self.current_ambient_temp_C = self._ambient_temp_eclipse_C # Initial value

    def update(self, dt: float, spacecraft_state: dict, action: dict | None = None):
        """Update temperatures based on heat flow, capacitance, and conductance."""
        temp_a_C, temp_b_C, heater_on = self.state
        is_sunlit = spacecraft_state.get('is_sunlit', True)

        # Update ambient temperature based on orbit state
        self.current_ambient_temp_C = self._ambient_temp_sunlit_C if is_sunlit else self._ambient_temp_eclipse_C

        # --- Determine Actual Heater State --- 
        # Apply control action (e.g., turn heater on/off)
        if action and 'heater_command' in action:
             if 'HeaterStuck' not in self.fault_status:
                heater_on = action['heater_command']
        # Apply persistent fault effects
        if self.fault_status == 'HeaterStuckOn':
            heater_on = 1.0
        elif self.fault_status == 'HeaterStuckOff':
            heater_on = 0.0
        # Store final heater status in state
        self.state[2] = heater_on 
        actual_heater_power_W = self.heater_power_w if heater_on > 0.5 else 0.0

        # --- Calculate Heat Flows (Watts) --- 
        # Between components A and B
        heat_flow_a_to_b = self.conductance_a_b_W_per_K * (temp_a_C - temp_b_C)
        # Between component A and Environment
        heat_flow_a_to_env = self.heat_coeff_a_env_W_per_K * (temp_a_C - self.current_ambient_temp_C)
        # Between component B and Environment
        heat_flow_b_to_env = self.heat_coeff_b_env_W_per_K * (temp_b_C - self.current_ambient_temp_C)
        
        # --- Calculate Net Heat Flow for Each Component (Watts = Joules/sec) --- 
        # Assume heater primarily heats component A
        Q_net_a = self.internal_heat_a_W + actual_heater_power_W - heat_flow_a_to_b - heat_flow_a_to_env
        Q_net_b = self.internal_heat_b_W + heat_flow_a_to_b - heat_flow_b_to_env

        # --- Calculate Temperature Change (delta_T = Q_net * dt / C) --- 
        delta_temp_a_C = (Q_net_a * dt) / self.capacitance_a_J_per_K
        delta_temp_b_C = (Q_net_b * dt) / self.capacitance_b_J_per_K

        # --- Update Temperatures --- 
        new_temp_a_C = temp_a_C + delta_temp_a_C
        new_temp_b_C = temp_b_C + delta_temp_b_C
        
        # Update state (temperatures)
        self.state[0] = new_temp_a_C
        self.state[1] = new_temp_b_C
        # self.state[2] (heater status) was updated earlier

    def get_telemetry(self) -> np.ndarray:
        temp_a, temp_b, heater_on = self.state
        reported_temp_a = temp_a
        reported_temp_b = temp_b
        reported_heater_on = heater_on

        # Check for persistent sensor fault first
        if self.fault_status == 'SensorAFailure':
            reported_temp_a = self._internal_fault_params.get('failed_value', -999.0)
        # Check for ACTIVE intermittent sensor fault
        elif self.is_intermittent_fault_active and self.active_intermittent_fault_type == 'SensorAIntermittent':
             reported_temp_a = self.active_intermittent_fault_params.get('failed_value', -888.0)
             
        # Add noise if needed
        # reported_temp_a += np.random.normal(0, 0.5)
        # reported_temp_b += np.random.normal(0, 0.5)
        
        return np.array([reported_temp_a, reported_temp_b, reported_heater_on], dtype=np.float32)

    def _apply_fault_effect(self):
        """Apply immediate effects of a TCS fault."""
        if self.fault_status == 'HeaterStuckOn':
            self.state[2] = 1.0 # Turn heater on immediately
            print(f"{self.name}: Heater stuck ON.")
        elif self.fault_status == 'HeaterStuckOff':
            self.state[2] = 0.0 # Turn heater off immediately
            print(f"{self.name}: Heater stuck OFF.")
        elif self.fault_status == 'SensorAFailure':
            # Effect is applied in get_telemetry
             print(f"{self.name}: Sensor A will report faulty values.")
        # Add other fault effects

    def _reset_persistent_fault_internals(self):
        pass # No internal vars affected by persistent TCS faults defined here
        
    def _reset_internal_state(self):
        # Reset ambient temp? No, that depends on orbit model.
        pass

# Add other subsystems like PropulsionSystem etc. as needed 
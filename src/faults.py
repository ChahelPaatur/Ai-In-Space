import random
import copy # Needed for deep copying fault info

# Define potential fault types (can be expanded significantly)
POSSIBLE_FAULTS = [
    # EPS Faults
    {'subsystem': 'EPS', 'type': 'SolarPanelDegradation', 'params': {'degradation_factor': 0.6}, 'intermittent': False},
    {'subsystem': 'EPS', 'type': 'BatteryCellFailure', 'params': {'capacity_reduction': 0.75}, 'intermittent': False},
    {'subsystem': 'EPS', 'type': 'PowerShortCircuit', 'params': {}, 'intermittent': False},

    # ADCS Faults
    {'subsystem': 'ADCS', 'type': 'GyroBias', 'params': {'bias_value': 0.01}, 'intermittent': False},
    {'subsystem': 'ADCS', 'type': 'ReactionWheelFriction', 'params': {'friction_increase': 1.5}, 'intermittent': False},
    {'subsystem': 'ADCS', 'type': 'GyroNoiseIncrease', 'params': {'noise_factor': 10.0}, 'intermittent': False},

    # TCS Faults
    {'subsystem': 'TCS', 'type': 'HeaterStuckOn', 'params': {}, 'intermittent': False},
    {'subsystem': 'TCS', 'type': 'HeaterStuckOff', 'params': {}, 'intermittent': False},
    {'subsystem': 'TCS', 'type': 'SensorAFailure', 'params': {'failed_value': -999.0}, 'intermittent': False},
    # Intermittent Example:
    {
        'subsystem': 'TCS', 
        'type': 'SensorAIntermittent', 
        'params': {'failed_value': -888.0}, 
        'intermittent': True, 
        'activation_prob_per_step': 0.05, # Chance to turn ON each step if OFF
        'deactivation_prob_per_step': 0.15  # Chance to turn OFF each step if ON
    },
]

class FaultInjector:
    """Handles the injection and state changes of persistent and intermittent faults."""
    def __init__(self, subsystems: dict, fault_probability: float = 0.01):
        """
        Args:
            subsystems (dict): Dictionary mapping subsystem names to subsystem objects.
            fault_probability (float): Probability of a new fault occurring at each step.
        """
        self.subsystems = subsystems
        self.fault_probability = fault_probability
        self.active_persistent_faults = {} 
        self.potential_intermittent_faults = [] 
        self.active_intermittent_fault_instances = []

    def step(self):
        """Called at each simulation step to potentially introduce new faults."""
        if random.random() < self.fault_probability:
            self.trigger_random_fault()

    def trigger_random_fault(self):
        """Selects and introduces a random fault (persistent or potential intermittent)."""
        if not POSSIBLE_FAULTS: return
        fault_info = copy.deepcopy(random.choice(POSSIBLE_FAULTS))
        subsystem_name = fault_info['subsystem']
        fault_type = fault_info['type']
        is_intermittent = fault_info.get('intermittent', False)

        if subsystem_name in self.subsystems:
            subsystem = self.subsystems[subsystem_name]
            if is_intermittent:
                # ... (introduce potential intermittent logic) ...
                already_potential = any(
                    f['subsystem'] == subsystem_name and f['type'] == fault_type 
                    for f in self.potential_intermittent_faults
                )
                if not already_potential:
                     print(f"\n--- Introducing POTENTIAL Intermittent Fault --- ")
                     print(f"  Type: {fault_type} on {subsystem_name}")
                     self.potential_intermittent_faults.append(fault_info)
                     print("--------------------------------------------\n")
            else:
                # Persistent fault: Apply if subsystem doesn't already have a persistent fault
                current_persistent_fault = self.active_persistent_faults.get(subsystem_name)
                if not current_persistent_fault:
                    print(f"\n--- Injecting Persistent Fault --- ")
                    print(f"  Type: {fault_type} on {subsystem_name}")
                    print(f"  Params: {fault_info['params']}")
                    subsystem.apply_fault(fault_info)
                    self.active_persistent_faults[subsystem_name] = fault_info
                    # Clear any potential *intermittent* faults on this subsystem
                    self.potential_intermittent_faults = [
                        f for f in self.potential_intermittent_faults 
                        if f['subsystem'] != subsystem_name
                    ]
                    # Ensure no intermittent faults are active either
                    self.active_intermittent_fault_instances = [
                        f for f in self.active_intermittent_fault_instances
                        if f['subsystem'] != subsystem_name
                    ]
                    # Call subsystem intermittent clear just in case
                    if hasattr(subsystem, 'deactivate_intermittent_fault'):
                         subsystem.deactivate_intermittent_fault(None)
                    print("----------------------------------\n")
        else:
            print(f"Warning: Tried to trigger fault for unknown subsystem: {subsystem_name}")

    def get_active_faults(self) -> list:
        """Returns the list of currently active faults."""
        # In a more complex sim, might check fault durations, clear resolved faults etc.
        # based on recovery actions or time limits.
        return self.active_faults 

    def get_active_faults_summary(self) -> dict:
        """Returns a summary of active faults."""
        summary = {
            "persistent": list(self.active_persistent_faults.values()),
            "intermittent_active": self.active_intermittent_fault_instances,
        }
        return summary

    def get_subsystem_status(self, subsystem_name: str) -> tuple[str, dict | None]:
        """Returns the status of a subsystem."""
        if subsystem_name in self.subsystems:
            subsystem = self.subsystems[subsystem_name]
            return subsystem.fault_status, subsystem.fault_info
        else:
            return 'Nominal', None

    def reset(self):
        """Resets the fault injector state and clears faults in subsystems."""
        print("Info: Resetting Fault Injector state.")
        # Also tell subsystems to clear persistent and intermittent state (important!)
        for sub_name, sub in self.subsystems.items():
             # Check if clear_fault_state method exists before calling
             if hasattr(sub, 'clear_fault_state') and callable(sub.clear_fault_state):
                 sub.clear_fault_state() # Clears persistent fault state
             # Check if deactivate_intermittent_fault method exists
             if hasattr(sub, 'deactivate_intermittent_fault') and callable(sub.deactivate_intermittent_fault):
                 sub.deactivate_intermittent_fault(None) # Clear intermittent flags
             
        self.active_persistent_faults = {}
        self.potential_intermittent_faults = []
        self.active_intermittent_fault_instances = [] 
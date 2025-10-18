import numpy as np

class FDIRMetrics:
    """
    Class for calculating and tracking FDIR performance metrics.
    
    # Paper reference: Section 3.5 "Metrics Framework" - This class implements the comprehensive
    # metrics framework described in the paper, including traditional metrics (MTTD, MTTR) and
    # the novel SFRI (Stability-Integrated Fault Recovery Index).
    
    This includes traditional metrics like MTTD (Mean Time To Detect) and MTTR 
    (Mean Time To Recover), as well as a novel SFRI (Stability-Integrated Fault 
    Recovery Index) that accounts for system stability impacts during recovery.
    """
    
    def __init__(self):
        """Initialize metrics tracking."""
        # Basic metrics
        self.fault_episodes = []
        self.detection_times = []
        self.recovery_times = []
        self.false_positives = 0
        self.total_actions = 0  # Track total actions to calculate false positive rate
        self.stability_impacts = []
        
        # Detection/Recovery actions
        self.recovery_actions = [1, 2, 3]  # RecoverEPS, RecoverADCS, RecoverTCS
        
        # SFRI calculation weights
        # Paper reference: Section 3.5 "Metrics Framework" - These weights are used in the
        # SFRI formula as described in the paper: 
        # "SFRI = 35×DetectionRate + 25×(1-MTTR/MaxSteps) + 10×StabilityScore - 30×FalsePositiveRate"
        self.detection_weight = 35.0      # Weight for detection rate (%)
        self.recovery_weight = 25.0       # Weight for recovery time (%)
        self.stability_weight = 10.0      # Weight for stability score (%)
        self.false_positive_weight = 30.0 # Weight for false positive rate (%)
        self.max_steps = 200.0            # Maximum steps for MTTR normalization
    
    def process_episode_log(self, episode_log, subsystem_fields=None):
        """
        Process a complete episode log to extract FDIR metrics.
        
        # Paper reference: Section 3.5 "Metrics Framework" - This function calculates all the
        # metrics discussed in the paper for a single episode, including MTTD, MTTR, detection
        # and recovery rates, false positives, and the SFRI score.
        
        Args:
            episode_log: List of step dictionaries from one episode
            subsystem_fields: Dict mapping subsystem names to their telemetry indices
                              Default uses standard indices if None
        
        Returns:
            dict: Dictionary of calculated metrics for this episode
        """
        # Default subsystem telemetry field mappings if not provided
        if subsystem_fields is None:
            subsystem_fields = {
                'EPS': {'soc': 0, 'voltage': 2},
                'ADCS': {'attitude_error': 11, 'angular_rates': slice(8, 11)},
                'TCS': {'temp_a': 12, 'temp_b': 13}
            }
        
        # Process fault episodes
        fault_episodes = self._extract_fault_episodes(episode_log)
        
        # Calculate stability impacts during fault episodes
        # Paper reference: Section 3.5 "Metrics Framework" - The stability impact component
        # of the SFRI metric that quantifies how much non-faulty subsystems were destabilized
        stability_impacts = self._calculate_stability_impacts(
            episode_log, fault_episodes, subsystem_fields)
        
        # Count false positives (recovery actions with no faults)
        # Paper reference: Section 4.1 "False Positives" - The paper compares false positive
        # rates across agent types, showing Rule-based had zero while DRL and Hybrid had more
        false_positives = self._count_false_positives(episode_log, fault_episodes)
        
        # Extract detection and recovery times
        # Paper reference: Section 4.1 "MTTD & MTTR" - The paper highlights the DRL agent's
        # 41% faster fault detection compared to the Rule-based approach
        ttd_values = []  # Time To Detect
        ttr_values = []  # Time To Recover
        
        for episode in fault_episodes:
            if episode['detection_step'] is not None:
                ttd = episode['detection_step'] - episode['start_step']
                ttd_values.append(ttd)
            
            if episode['recovery_step'] is not None:
                ttr = episode['recovery_step'] - episode['start_step']
                ttr_values.append(ttr)
        
        # FIXED: Count total actions for this episode only (not cumulative)
        total_episode_actions = len([step for step in episode_log if 'action' in step])
        recovery_actions_count = sum(1 for step in episode_log if step.get('action') in self.recovery_actions)
        
        # Calculate SFRI for this episode with correct per-episode actions
        # Paper reference: Section 3.5 "Metrics Framework" - The novel integrated metric
        # that combines detection, recovery, stability, and false positive considerations
        sfri = self._calculate_sfri_episode(
            ttd_values, ttr_values, stability_impacts, 
            false_positives, len(fault_episodes), total_episode_actions
        )
        
        # Update overall metrics
        self.fault_episodes.extend(fault_episodes)
        self.detection_times.extend(ttd_values)
        self.recovery_times.extend(ttr_values)
        self.false_positives += false_positives
        self.total_actions += total_episode_actions  # Still track cumulative for aggregate
        self.stability_impacts.extend(stability_impacts)
        
        # Return episode metrics
        # Paper reference: Section 4.1 - These metrics are directly used in the figures
        # and analysis presented in the "Aggregate Performance Metrics" section
        return {
            'mttd': np.mean(ttd_values) if ttd_values else float('inf'),
            'mttr': np.mean(ttr_values) if ttr_values else float('inf'),
            'detection_rate': len(ttd_values) / len(fault_episodes) if fault_episodes else 1.0,
            'recovery_rate': len(ttr_values) / len(fault_episodes) if fault_episodes else 1.0,
            'false_positives': false_positives,
            'fault_episodes': len(fault_episodes),
            'stability_impact': np.mean(stability_impacts) if stability_impacts else 0.0,
            'sfri': sfri
        }
    
    def get_aggregate_metrics(self):
        """
        Get aggregate metrics across all processed episodes.
        
        # Paper reference: Section 4.1 - The metrics returned by this function are analyzed
        # in the "Aggregate Performance Metrics" section and visualized in Figures 1-5.
        """
        # Calculate detection and recovery rates
        # Paper reference: Section 4.1 "Detection & Recovery Rates" - The paper highlights the
        # Hybrid agent's 100% detection rate compared to DRL (100%) and Rule-based (33.7%)
        total_faults = len(self.fault_episodes)
        detection_rate = len(self.detection_times) / total_faults if total_faults > 0 else 1.0
        recovery_rate = len(self.recovery_times) / total_faults if total_faults > 0 else 1.0
        
        # Calculate MTTD and MTTR
        # Paper reference: Section 4.1 "MTTD & MTTR" - The paper notes the DRL agent's
        # significantly faster fault detection (MTTD of 21.77 vs. 36.65 steps)
        mttd = np.mean(self.detection_times) if self.detection_times else float('inf')
        mttr = np.mean(self.recovery_times) if self.recovery_times else float('inf')
        
        # Calculate average stability impact
        avg_stability_impact = np.mean(self.stability_impacts) if self.stability_impacts else 0.0
        
        # Calculate overall SFRI
        # Paper reference: Section 4.1 "SFRI Metric" - The Hybrid agent achieved the highest
        # score (40.0/100), followed by Rule-based (38.5/100) and DRL (37.9/100)
        sfri = self._calculate_sfri(
            self.detection_times, self.recovery_times, 
            self.stability_impacts, self.false_positives, total_faults
        )
        
        return {
            'mttd': mttd,
            'mttr': mttr,
            'detection_rate': detection_rate,
            'recovery_rate': recovery_rate,
            'false_positives': self.false_positives,
            'fault_episodes': total_faults,
            'stability_impact': avg_stability_impact,
            'sfri': sfri
        }
    
    def _extract_fault_episodes(self, episode_log, fault_field='persistent_faults'):
        """
        Extract fault episodes from the episode log.
        
        # Paper reference: Section 4.2 "Dynamic Response Characteristics" - This function
        # identifies the fault episodes described in the paper, such as the HeaterStuckOff
        # fault in Episode 84 that showed different detection and response patterns
        # across agent types.
        
        Args:
            episode_log: List of step dictionaries
            fault_field: Field name containing fault information
            
        Returns:
            list: List of fault episode dictionaries
        """
        fault_episodes = []
        current_episode = None
        
        for step_idx, step_data in enumerate(episode_log):
            faults = step_data.get(fault_field, [])
            action = step_data.get('action')
            
            # Case 1: New fault detected, no current tracking
            if faults and current_episode is None:
                current_episode = {
                    'start_step': step_idx,
                    'faults': faults.copy(),
                    'detection_step': None,
                    'recovery_step': None,
                    'actions_taken': [],
                    'affected_subsystems': self._get_affected_subsystems(faults)
                }
            
            # Case 2: Ongoing fault episode
            elif faults and current_episode is not None:
                # Update fault list if needed
                for fault in faults:
                    if fault not in current_episode['faults']:
                        current_episode['faults'].append(fault)
                        # Update affected subsystems
                        affected = self._get_affected_subsystems([fault])
                        for subsys in affected:
                            if subsys not in current_episode['affected_subsystems']:
                                current_episode['affected_subsystems'].append(subsys)
                
                # Check if this is a recovery action
                if action in self.recovery_actions and current_episode['detection_step'] is None:
                    current_episode['detection_step'] = step_idx
                
                # Record all actions for analysis
                current_episode['actions_taken'].append(action)
                
                # If we previously detected and now faults are resolved, mark recovery
                next_step = episode_log[step_idx + 1] if step_idx + 1 < len(episode_log) else None
                if next_step and not next_step.get(fault_field, []) and current_episode['recovery_step'] is None:
                    current_episode['recovery_step'] = step_idx + 1
                    fault_episodes.append(current_episode)
                    current_episode = None
            
            # Case 3: No faults, but we were tracking an episode
            elif not faults and current_episode is not None:
                # End of fault without recovery action
                if current_episode['recovery_step'] is None:
                    current_episode['recovery_step'] = step_idx
                fault_episodes.append(current_episode)
                current_episode = None
        
        # Handle any incomplete episode at the end
        if current_episode is not None:
            current_episode['recovery_step'] = len(episode_log)  # Mark as unrecovered
            fault_episodes.append(current_episode)
        
        return fault_episodes
    
    def _get_affected_subsystems(self, faults):
        """
        Determine which subsystems are affected by the given faults.
        
        Args:
            faults: List of fault names
            
        Returns:
            list: List of affected subsystem names
        """
        affected = []
        for fault in faults:
            # Extract subsystem from fault name
            if 'EPS' in fault or 'Battery' in fault or 'Solar' in fault:
                if 'EPS' not in affected:
                    affected.append('EPS')
            elif 'ADCS' in fault or 'Gyro' in fault or 'Wheel' in fault:
                if 'ADCS' not in affected:
                    affected.append('ADCS')
            elif 'TCS' in fault or 'Heater' in fault or 'Temp' in fault:
                if 'TCS' not in affected:
                    affected.append('TCS')
        
        return affected
    
    def _calculate_stability_impacts(self, episode_log, fault_episodes, subsystem_fields):
        """
        Calculate stability impact metrics for each fault episode.
        
        # Paper reference: Section 3.5 "Metrics Framework" - This implements the stability
        # component of the SFRI metric, measuring how much other subsystems were destabilized
        # during fault detection and recovery. This addresses the oscillatory behaviors
        # described in Section 4.2 "Dynamic Response Characteristics".
        
        This measures how much other subsystems were destabilized during 
        fault detection and recovery.
        
        Args:
            episode_log: List of step dictionaries
            fault_episodes: List of fault episode dictionaries
            subsystem_fields: Dict mapping subsystem telemetry fields
            
        Returns:
            list: List of stability impact scores (higher is worse)
        """
        stability_impacts = []
        
        for episode in fault_episodes:
            # Get the primary affected subsystem
            primary_subsystems = episode['affected_subsystems']
            
            # Skip if we can't determine the affected subsystem
            if not primary_subsystems:
                continue
            
            # Define start and end steps for analysis
            start_step = episode['start_step']
            end_step = episode['recovery_step'] or len(episode_log)
            
            # Calculate stability impacts on other subsystems
            stability_impact = 0
            
            # Check EPS stability if not the primary affected subsystem
            if 'EPS' not in primary_subsystems:
                # Calculate SoC and voltage stability before and during fault
                soc_field = subsystem_fields['EPS']['soc']
                voltage_field = subsystem_fields['EPS']['voltage']
                
                # Analyze observations before fault
                pre_fault_steps = max(0, start_step - 10), start_step
                pre_fault_soc = [
                    episode_log[i]['observation'][soc_field] 
                    for i in range(*pre_fault_steps)
                    if i < len(episode_log)
                ]
                pre_fault_voltage = [
                    episode_log[i]['observation'][voltage_field] 
                    for i in range(*pre_fault_steps)
                    if i < len(episode_log)
                ]
                
                # Analyze observations during fault
                during_fault_soc = [
                    episode_log[i]['observation'][soc_field] 
                    for i in range(start_step, end_step)
                    if i < len(episode_log)
                ]
                during_fault_voltage = [
                    episode_log[i]['observation'][voltage_field] 
                    for i in range(start_step, end_step)
                    if i < len(episode_log)
                ]
                
                # Calculate stability metrics
                if pre_fault_soc and during_fault_soc:
                    soc_instability = (
                        np.std(during_fault_soc) - np.std(pre_fault_soc)
                    ) / max(0.01, np.mean(pre_fault_soc))
                    stability_impact += max(0, soc_instability)
                
                if pre_fault_voltage and during_fault_voltage:
                    voltage_instability = (
                        np.std(during_fault_voltage) - np.std(pre_fault_voltage)
                    ) / max(0.01, np.mean(pre_fault_voltage))
                    stability_impact += max(0, voltage_instability)
            
            # Check ADCS stability if not the primary affected subsystem
            if 'ADCS' not in primary_subsystems:
                # Calculate attitude error and angular rates stability
                att_err_field = subsystem_fields['ADCS']['attitude_error']
                rates_field = subsystem_fields['ADCS']['angular_rates']
                
                # Analyze observations before fault
                pre_fault_steps = max(0, start_step - 10), start_step
                pre_fault_att_err = [
                    episode_log[i]['observation'][att_err_field] 
                    for i in range(*pre_fault_steps)
                    if i < len(episode_log)
                ]
                
                # Angular rates require special handling (vector)
                pre_fault_rates = []
                for i in range(*pre_fault_steps):
                    if i < len(episode_log):
                        rates = episode_log[i]['observation'][rates_field]
                        pre_fault_rates.append(np.linalg.norm(rates))
                
                # Analyze observations during fault
                during_fault_att_err = [
                    episode_log[i]['observation'][att_err_field] 
                    for i in range(start_step, end_step)
                    if i < len(episode_log)
                ]
                
                during_fault_rates = []
                for i in range(start_step, end_step):
                    if i < len(episode_log):
                        rates = episode_log[i]['observation'][rates_field]
                        during_fault_rates.append(np.linalg.norm(rates))
                
                # Calculate stability metrics
                if pre_fault_att_err and during_fault_att_err:
                    att_err_instability = (
                        np.mean(during_fault_att_err) - np.mean(pre_fault_att_err)
                    ) / max(0.01, np.mean(pre_fault_att_err))
                    stability_impact += max(0, att_err_instability)
                
                if pre_fault_rates and during_fault_rates:
                    rates_instability = (
                        np.mean(during_fault_rates) - np.mean(pre_fault_rates)
                    ) / max(0.01, np.mean(pre_fault_rates))
                    stability_impact += max(0, rates_instability)
            
            # Check TCS stability if not the primary affected subsystem
            if 'TCS' not in primary_subsystems:
                # Calculate temperature stability
                temp_a_field = subsystem_fields['TCS']['temp_a']
                temp_b_field = subsystem_fields['TCS']['temp_b']
                
                # Analyze observations before fault
                pre_fault_steps = max(0, start_step - 10), start_step
                pre_fault_temp_a = [
                    episode_log[i]['observation'][temp_a_field] 
                    for i in range(*pre_fault_steps)
                    if i < len(episode_log)
                ]
                pre_fault_temp_b = [
                    episode_log[i]['observation'][temp_b_field] 
                    for i in range(*pre_fault_steps)
                    if i < len(episode_log)
                ]
                
                # Analyze observations during fault
                during_fault_temp_a = [
                    episode_log[i]['observation'][temp_a_field] 
                    for i in range(start_step, end_step)
                    if i < len(episode_log)
                ]
                during_fault_temp_b = [
                    episode_log[i]['observation'][temp_b_field] 
                    for i in range(start_step, end_step)
                    if i < len(episode_log)
                ]
                
                # Calculate stability metrics
                if pre_fault_temp_a and during_fault_temp_a:
                    temp_a_instability = (
                        np.std(during_fault_temp_a) - np.std(pre_fault_temp_a)
                    ) / max(0.01, abs(np.mean(pre_fault_temp_a)))
                    stability_impact += max(0, temp_a_instability)
                
                if pre_fault_temp_b and during_fault_temp_b:
                    temp_b_instability = (
                        np.std(during_fault_temp_b) - np.std(pre_fault_temp_b)
                    ) / max(0.01, abs(np.mean(pre_fault_temp_b)))
                    stability_impact += max(0, temp_b_instability)
            
            # Normalize and add to results
            stability_impact = min(1.0, stability_impact / 2.0)  # Cap at 1.0
            stability_impacts.append(stability_impact)
        
        return stability_impacts
    
    def _count_false_positives(self, episode_log, fault_episodes):
        """
        Count false positive recovery actions.
        
        # Paper reference: Section 4.1 "False Positives" - This function calculates the
        # false positive counts discussed in the paper and visualized in Figure 3, where
        # the Rule-based agent showed zero false positives while DRL and Hybrid agents
        # triggered unnecessary recoveries.
        
        Args:
            episode_log: List of step dictionaries
            fault_episodes: List of fault episode dictionaries
            
        Returns:
            int: Number of false positive recovery actions
        """
        false_positives = 0
        
        # Create a set of steps that are part of a fault episode
        fault_steps = set()
        for episode in fault_episodes:
            start = episode['start_step']
            end = episode['recovery_step'] or len(episode_log)
            # Include a grace period of 5 steps after recovery
            for step in range(start, min(end + 5, len(episode_log))):
                fault_steps.add(step)
        
        # Count recovery actions outside fault episodes
        for step_idx, step_data in enumerate(episode_log):
            if step_idx not in fault_steps and step_data.get('action') in self.recovery_actions:
                false_positives += 1
        
        return false_positives
    
    def _calculate_sfri_episode(self, detection_times, recovery_times, stability_impacts, 
                               false_positives, total_faults, total_episode_actions):
        """
        Calculate the Stability-Integrated Fault Recovery Index (SFRI) for a single episode.
        
        # Paper reference: Section 3.5 "Metrics Framework" - This implements the SFRI
        # metric with weights:
        # "SFRI = 35×DetectionRate + 25×(1-MTTR/MaxSteps) + 10×StabilityScore - 30×FalsePositiveRate"
        # Maximum possible score: 35 + 25 + 10 - 0 = 70.0 points
        
        Args:
            detection_times: List of time-to-detect values
            recovery_times: List of time-to-recover values
            stability_impacts: List of stability impact scores
            false_positives: Number of false positive recovery actions in this episode
            total_faults: Number of fault episodes
            total_episode_actions: Total actions taken in this specific episode
            
        Returns:
            float: SFRI score (higher is better, max 70.0)
        """
        if total_faults == 0:
            return 0.0
            
        # Detection Rate (35% weight)
        detection_rate = len(detection_times) / total_faults
        detection_component = 35.0 * detection_rate
        
        # MTTR Component (25% weight)
        if recovery_times:
            avg_mttr = sum(recovery_times) / len(recovery_times)
            recovery_score = max(0.0, 1.0 - (avg_mttr / self.max_steps))
        else:
            recovery_score = 0.0
        recovery_component = 25.0 * recovery_score
        
        # Stability Component (10% weight)
        if stability_impacts:
            avg_stability = sum(stability_impacts) / len(stability_impacts)
            stability_score = max(0.0, 1.0 - avg_stability)
        else:
            stability_score = 1.0
        stability_component = 10.0 * stability_score
        
        # False Positive Penalty (30% weight)
        if total_episode_actions > 0:
            false_positive_rate = false_positives / total_episode_actions
        else:
            false_positive_rate = 0.0
        false_positive_penalty = 30.0 * false_positive_rate
        
        # Calculate final SFRI (max possible: 70.0)
        sfri = detection_component + recovery_component + stability_component - false_positive_penalty
        
        return max(0.0, sfri)  # Ensure non-negative

    def _calculate_sfri(self, detection_times, recovery_times, stability_impacts, 
                       false_positives, total_faults):
        """
        Calculate the Stability-Integrated Fault Recovery Index (SFRI).
        
        # Paper reference: Section 3.5 "Metrics Framework" - This implements the SFRI
        # metric with weights:
        # "SFRI = 35×DetectionRate + 25×(1-MTTR/MaxSteps) + 10×StabilityScore - 30×FalsePositiveRate"
        # Maximum possible score: 35 + 25 + 10 - 0 = 70.0 points
        
        Args:
            detection_times: List of time-to-detect values
            recovery_times: List of time-to-recover values
            stability_impacts: List of stability impact scores
            false_positives: Number of false positive recovery actions
            total_faults: Total number of fault episodes
            
        Returns:
            float: SFRI score (higher is better, max 70.0)
        """
        if total_faults == 0:
            return 0.0
            
        # Detection Rate (35% weight)
        detection_rate = len(detection_times) / total_faults
        detection_component = 35.0 * detection_rate
        
        # MTTR Component (25% weight)
        if recovery_times:
            avg_mttr = sum(recovery_times) / len(recovery_times)
            recovery_score = max(0.0, 1.0 - (avg_mttr / self.max_steps))
        else:
            recovery_score = 0.0
        recovery_component = 25.0 * recovery_score
        
        # Stability Component (10% weight)
        if stability_impacts:
            avg_stability = sum(stability_impacts) / len(stability_impacts)
            stability_score = max(0.0, 1.0 - avg_stability)
        else:
            stability_score = 1.0
        stability_component = 10.0 * stability_score
        
        # False Positive Penalty (30% weight)
        if self.total_actions > 0:
            false_positive_rate = false_positives / self.total_actions
        else:
            false_positive_rate = 0.0
        false_positive_penalty = 30.0 * false_positive_rate
        
        # Calculate final SFRI (max possible: 70.0)
        sfri = detection_component + recovery_component + stability_component - false_positive_penalty
        
        return max(0.0, sfri)  # Ensure non-negative 
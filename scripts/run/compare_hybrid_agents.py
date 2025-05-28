import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import argparse
import random

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.environment import SpacecraftEnv

# Mock agents for testing
class MockRuleBasedFDIR:
    """Simple rule-based FDIR agent."""
    
    def __init__(self):
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
        
        self.thresholds = {
            'eps_voltage_low': 0.3,
            'adcs_error_high': 0.7,
            'tcs_temp_high': 0.8,
            'tcs_temp_low': 0.2
        }
        
        self.recovery_actions = [1, 2, 3]
        self.safety_critical_actions = [1, 2, 3, 7]
    
    def get_action(self, observation, info):
        """Simple threshold-based decision making."""
        # Check thresholds
        eps_value = observation[0]
        adcs_value = observation[4]
        tcs_value = observation[8]
        
        action = 0  # Default to NO_OP
        
        if eps_value < self.thresholds['eps_voltage_low']:
            action = 1  # RECOVER_EPS
        elif adcs_value > self.thresholds['adcs_error_high']:
            action = 2  # RECOVER_ADCS
        elif tcs_value > self.thresholds['tcs_temp_high'] or tcs_value < self.thresholds['tcs_temp_low']:
            action = 3  # RECOVER_TCS
        
        return action, {"rule_action": action}

class MockHybridFDIRAgent:
    """
    Mock Hybrid FDIR agent that combines rule-based and randomized "DRL-like" decisions.
    Has high false positive rate.
    """
    
    def __init__(self):
        self.rule_agent = MockRuleBasedFDIR()
        self.ACTION_MAP = self.rule_agent.ACTION_MAP
        self.recovery_actions = self.rule_agent.recovery_actions
        self.safety_critical_actions = self.rule_agent.safety_critical_actions
        
        self.decision_source_map = {
            "rule": "Rule-Based",
            "drl": "DRL",
            "rule_override": "Rule (Safety Override)",
            "drl_confident": "DRL (High Confidence)",
            "cooldown": "Rule (Cooldown Period)"
        }
        
        self.recovery_cooldown = 0
        self.cooldown_period = 10
        self.decision_source_history = []
        self.override_count = 0
        self.rule_count = 0
    
    def get_action(self, observation, info):
        """Hybrid decision making with high false positive rate."""
        rule_action = self.rule_agent.get_action(observation, info)[0]
        
        # Simulate DRL with randomized action and confidence
        drl_action = random.randint(0, 8)
        drl_confidence = random.random()
        
        # Handle recovery cooldown
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
            decision_source = "cooldown"
            
            if rule_action in self.safety_critical_actions:
                final_action = rule_action
            else:
                final_action = self.ACTION_MAP["NO_OP"]
        else:
            # Regular decision logic
            decision_source = "rule"  # Default
            final_action = rule_action
            
            # Case 1: Rule-based system chooses safety-critical action
            if rule_action in self.safety_critical_actions:
                decision_source = "rule_override"
                final_action = rule_action
                
                if rule_action in self.recovery_actions:
                    self.recovery_cooldown = self.cooldown_period
            
            # Case 2: DRL is confident
            elif drl_confidence >= 0.7:
                # High chance (40%) of false positive for recovery actions to simulate the issue
                if drl_action in self.recovery_actions and random.random() < 0.4:
                    decision_source = "drl_confident"
                    final_action = drl_action
                    self.recovery_cooldown = self.cooldown_period
                else:
                    decision_source = "drl"
                    final_action = drl_action
        
        # Update metrics
        self.decision_source_history.append(decision_source)
        if "drl" in decision_source:
            self.override_count += 1
        else:
            self.rule_count += 1
        
        # Prepare decision info
        decision_info = {
            "rule_action": rule_action,
            "drl_action": drl_action,
            "drl_confidence": drl_confidence,
            "confidence_threshold": 0.7,
            "cooldown_remaining": self.recovery_cooldown,
            "decision_source": self.decision_source_map.get(decision_source, decision_source),
            "override_count": self.override_count,
            "rule_count": self.rule_count
        }
        
        return final_action, decision_info

class MockEnhancedHybridFDIRAgent:
    """
    Mock Enhanced Hybrid FDIR agent with two-stage detection system.
    Significantly reduces false positives.
    """
    
    def __init__(self):
        self.rule_agent = MockRuleBasedFDIR()
        self.ACTION_MAP = self.rule_agent.ACTION_MAP
        self.recovery_actions = self.rule_agent.recovery_actions
        self.safety_critical_actions = self.rule_agent.safety_critical_actions
        
        # Two-stage detection system parameters
        self.anomaly_history = []
        self.anomaly_scores = {'EPS': 0, 'ADCS': 0, 'TCS': 0}
        
        self.decision_source_map = {
            "rule": "Rule-Based",
            "drl": "DRL",
            "rule_override": "Rule (Safety Override)",
            "drl_confident": "DRL (High Confidence)",
            "cooldown": "Rule (Cooldown Period)",
            "two_stage": "Two-Stage Validation",
            "blocked": "Blocked by Validation"
        }
        
        self.recovery_cooldown = 0
        self.cooldown_period = 10
        self.decision_source_history = []
        self.override_count = 0
        self.rule_count = 0
        self.false_positive_prevented = 0
    
    def get_action(self, observation, info):
        """Two-stage hybrid decision making with lower false positive rate."""
        rule_action = self.rule_agent.get_action(observation, info)[0]
        
        # Simulate DRL with randomized action and confidence
        drl_action = random.randint(0, 8)
        drl_confidence = random.random()
        
        # First stage: Detect potential anomalies
        anomalies = {}
        
        # Rule-based anomaly detection
        if observation[0] < self.rule_agent.thresholds['eps_voltage_low']:
            anomalies['EPS'] = 1.0
        if observation[4] > self.rule_agent.thresholds['adcs_error_high']:
            anomalies['ADCS'] = 1.0
        if (observation[8] > self.rule_agent.thresholds['tcs_temp_high'] or 
            observation[8] < self.rule_agent.thresholds['tcs_temp_low']):
            anomalies['TCS'] = 1.0
        
        # Simulate DRL anomaly detection
        if drl_action in self.recovery_actions:
            subsys = ['EPS', 'ADCS', 'TCS'][drl_action - 1]
            anomalies[subsys] = anomalies.get(subsys, 0) + 0.8 * drl_confidence
        
        # Second stage: Validate anomalies
        # Update anomaly history (simplified)
        self.anomaly_history.append(anomalies)
        if len(self.anomaly_history) > 5:
            self.anomaly_history.pop(0)
        
        # Update anomaly scores with decay
        validated_anomalies = {}
        for subsys in ['EPS', 'ADCS', 'TCS']:
            # Apply decay
            self.anomaly_scores[subsys] *= 0.8
            
            # Add current score
            self.anomaly_scores[subsys] += anomalies.get(subsys, 0.0)
            
            # Cap at 5.0
            self.anomaly_scores[subsys] = min(5.0, self.anomaly_scores[subsys])
            
            # Check for validation threshold
            if self.anomaly_scores[subsys] >= 0.8:
                # Count how many recent observations had this anomaly
                consistent_count = sum(1 for past in self.anomaly_history if subsys in past)
                
                # Validate only if consistently detected
                if consistent_count >= 3:
                    validated_anomalies[subsys] = self.anomaly_scores[subsys]
        
        # Handle recovery cooldown
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
            decision_source = "cooldown"
            
            if rule_action in self.safety_critical_actions:
                final_action = rule_action
            else:
                final_action = self.ACTION_MAP["NO_OP"]
        else:
            # Check if rule-based agent suggests recovery
            rule_suggests_recovery = rule_action in self.recovery_actions
            
            # Check if validated anomalies match the recovery action
            validated_recovery = False
            if rule_suggests_recovery:
                subsys = ['EPS', 'ADCS', 'TCS'][rule_action - 1]
                if subsys in validated_anomalies:
                    validated_recovery = True
            
            # Regular decision logic
            decision_source = "rule"  # Default
            final_action = rule_action
            
            # Case 1: Rule-based system chooses safety-critical action
            if rule_action in self.safety_critical_actions:
                if validated_recovery or rule_action == self.ACTION_MAP["ENTER_SAFE_MODE"]:
                    decision_source = "two_stage"
                    final_action = rule_action
                    
                    if rule_action in self.recovery_actions:
                        self.recovery_cooldown = self.cooldown_period
                else:
                    # Block potential false positive
                    decision_source = "blocked"
                    final_action = self.ACTION_MAP["NO_OP"]
                    self.false_positive_prevented += 1
            
            # Case 2: DRL is confident
            elif drl_confidence >= 0.7:
                if drl_action in self.recovery_actions:
                    # Map to subsystem
                    subsys = ['EPS', 'ADCS', 'TCS'][drl_action - 1]
                    
                    # Verify with validation
                    if subsys in validated_anomalies:
                        decision_source = "drl_confident"
                        final_action = drl_action
                        self.recovery_cooldown = self.cooldown_period
                    else:
                        # Block potential false positive
                        decision_source = "blocked"
                        final_action = self.ACTION_MAP["NO_OP"]
                        self.false_positive_prevented += 1
                else:
                    # For non-recovery actions, trust DRL
                    decision_source = "drl"
                    final_action = drl_action
        
        # Update metrics
        self.decision_source_history.append(decision_source)
        if "drl" in decision_source:
            self.override_count += 1
        else:
            self.rule_count += 1
        
        # Prepare decision info
        decision_info = {
            "rule_action": rule_action,
            "drl_action": drl_action,
            "drl_confidence": drl_confidence,
            "confidence_threshold": 0.7,
            "cooldown_remaining": self.recovery_cooldown,
            "decision_source": self.decision_source_map.get(decision_source, decision_source),
            "override_count": self.override_count,
            "rule_count": self.rule_count,
            "detected_anomalies": anomalies,
            "validated_anomalies": validated_anomalies,
            "false_positive_prevented": self.false_positive_prevented
        }
        
        return final_action, decision_info

def run_episode(env, agent, max_steps=200, render=False, seed=None):
    """Run a single episode with the given agent and environment."""
    obs, info = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0
    steps = 0
    
    # Tracking metrics
    metrics = {
        "rewards": [],
        "actions": [],
        "decision_sources": [],
        "false_positives": 0,
        "recovery_actions": 0,
        "faults": [],
        "detected_faults": 0,
        "successful_recoveries": 0,
        "fault_detection_times": [],
        "fault_recovery_times": [],
        "stability_scores": []
    }
    
    # Track active faults and their detection/recovery
    active_faults = {}
    
    while not done and not truncated and steps < max_steps:
        # Get action from agent
        action, decision_info = agent.get_action(obs, info)
        
        # Execute action in environment
        next_obs, reward, done, truncated, next_info = env.step(action)
        
        # Track metrics
        metrics["rewards"].append(reward)
        metrics["actions"].append(action)
        
        # Track stability (simplified - assuming values close to 0.5 are stable)
        stability = 1.0 - (np.mean(np.abs(next_obs - 0.5)) / 0.5)
        metrics["stability_scores"].append(max(0.0, min(1.0, stability)))
        
        if "decision_source" in decision_info:
            metrics["decision_sources"].append(decision_info["decision_source"])
        
        # Track new faults
        if "fault_injected" in next_info and next_info["fault_injected"]:
            fault_info = {
                "step": steps,
                "type": next_info.get("fault_type", "Unknown"),
                "subsystem": next_info.get("fault_subsystem", "Unknown"),
                "detected": False,
                "detection_step": None,
                "recovered": False,
                "recovery_step": None
            }
            metrics["faults"].append(fault_info)
            fault_id = len(metrics["faults"]) - 1
            active_faults[fault_id] = fault_info
        
        # Track fault detection and recovery
        is_recovery_action = action in agent.recovery_actions
        
        # If it's a recovery action, check if it corresponds to an active fault
        if is_recovery_action:
            metrics["recovery_actions"] += 1
            
            # Map action to subsystem
            target_subsystem = None
            if action == 1:
                target_subsystem = "EPS"
            elif action == 2:
                target_subsystem = "ADCS"
            elif action == 3:
                target_subsystem = "TCS"
            
            # Check if this matches any active faults
            matched_fault = False
            for fault_id, fault in active_faults.items():
                if fault["subsystem"] == target_subsystem:
                    matched_fault = True
                    
                    # Mark as detected if not already
                    if not fault["detected"]:
                        fault["detected"] = True
                        fault["detection_step"] = steps
                        metrics["detected_faults"] += 1
                        metrics["fault_detection_times"].append(steps - fault["step"])
                    
                    # Check if fault is resolved in next state
                    # Simplified check - in a real system would need proper resolution check
                    if not env.has_active_fault():
                        fault["recovered"] = True
                        fault["recovery_step"] = steps
                        metrics["successful_recoveries"] += 1
                        metrics["fault_recovery_times"].append(steps - fault["step"])
                        active_faults.pop(fault_id, None)
                    break
            
            # If no matching fault, it's a false positive
            if not matched_fault and not env.has_active_fault():
                metrics["false_positives"] += 1
        
        # Update for next step
        obs = next_obs
        info = next_info
        total_reward += reward
        steps += 1
        
        if render:
            env.render()
    
    # Calculate additional metrics
    metrics["total_reward"] = total_reward
    metrics["steps"] = steps
    metrics["episode_complete"] = steps >= max_steps and not done and not truncated
    
    # Handle undetected/unrecovered faults
    for fault in metrics["faults"]:
        if not fault["detected"]:
            metrics["fault_detection_times"].append(max_steps)  # Max penalty
        if not fault["recovered"]:
            metrics["fault_recovery_times"].append(max_steps)  # Max penalty
    
    return metrics

def calculate_sfri(metrics, max_steps=200):
    """
    Calculate the Stability Fault Recovery Index (SFRI).
    
    # Paper reference: Section 3.5 "Evaluation Methodology" - This implements
    # the SFRI formula with the weights described in the paper.
    
    SFRI = 35 × (DetectionRate) + 25 × (1 - MTTR/MaxSteps) + 10 × (StabilityScore) - 30 × (FalsePositiveRate)
    """
    # Calculate detection rate
    total_faults = len(metrics["faults"])
    detection_rate = metrics["detected_faults"] / total_faults if total_faults > 0 else 1.0
    
    # Calculate recovery rate and times
    recovery_rate = metrics["successful_recoveries"] / total_faults if total_faults > 0 else 1.0
    mttr = np.mean(metrics["fault_recovery_times"]) if metrics["fault_recovery_times"] else max_steps
    
    # Calculate stability score
    stability_score = np.mean(metrics["stability_scores"]) if metrics["stability_scores"] else 0.5
    
    # Calculate false positive rate
    recovery_actions = metrics["recovery_actions"]
    false_positive_rate = metrics["false_positives"] / recovery_actions if recovery_actions > 0 else 0.0
    
    # Calculate SFRI components
    detection_component = 35.0 * detection_rate
    recovery_component = 25.0 * (1.0 - mttr / max_steps)
    stability_component = 10.0 * stability_score
    false_positive_component = 30.0 * false_positive_rate
    
    # Calculate final SFRI
    sfri = detection_component + recovery_component + stability_component - false_positive_component
    
    # Ensure it's within the 0-100 range
    sfri = max(0.0, min(100.0, sfri))
    
    return sfri

def evaluate_agent(env_config, agent, num_episodes=100, max_steps=200, seed=None):
    """Evaluate an agent over multiple episodes."""
    # Initialize environment
    env = SpacecraftEnv(**env_config)
    
    # Set seeds if provided
    if seed is not None:
        np.random.seed(seed)
        env.seed(seed)
    
    # Run episodes
    episode_metrics = []
    
    for episode in tqdm(range(num_episodes), desc=f"Evaluating {agent.__class__.__name__}"):
        # Generate a seed for this episode
        episode_seed = np.random.randint(0, 10000) if seed is None else seed + episode
        
        # Run episode
        metrics = run_episode(env, agent, max_steps=max_steps, seed=episode_seed)
        episode_metrics.append(metrics)
    
    # Calculate aggregate metrics
    aggregate_metrics = calculate_aggregate_metrics(episode_metrics, max_steps)
    
    return {
        "episode_metrics": episode_metrics,
        "aggregate_metrics": aggregate_metrics
    }

def calculate_aggregate_metrics(episode_metrics, max_steps=200):
    """Calculate aggregate metrics across multiple episodes."""
    # Initialize aggregate metrics
    aggregate = {
        "mean_reward": 0,
        "std_reward": 0,
        "total_false_positives": 0,
        "false_positive_rate": 0,
        "recovery_action_count": 0,
        "action_distribution": {},
        "decision_source_distribution": {},
        "detection_rate": 0,
        "recovery_rate": 0,
        "mttd": 0,  # Mean Time To Detect
        "mttr": 0,  # Mean Time To Recover
        "sfri": 0   # Stability Fault Recovery Index
    }
    
    # Collect rewards
    rewards = [m["total_reward"] for m in episode_metrics]
    aggregate["mean_reward"] = np.mean(rewards)
    aggregate["std_reward"] = np.std(rewards)
    
    # Collect false positives and recovery actions
    aggregate["total_false_positives"] = sum(m["false_positives"] for m in episode_metrics)
    aggregate["recovery_action_count"] = sum(m["recovery_actions"] for m in episode_metrics)
    
    if aggregate["recovery_action_count"] > 0:
        aggregate["false_positive_rate"] = aggregate["total_false_positives"] / aggregate["recovery_action_count"]
    
    # Calculate detection and recovery rates
    total_faults = sum(len(m["faults"]) for m in episode_metrics)
    detected_faults = sum(m["detected_faults"] for m in episode_metrics)
    recovered_faults = sum(m["successful_recoveries"] for m in episode_metrics)
    
    if total_faults > 0:
        aggregate["detection_rate"] = detected_faults / total_faults
        aggregate["recovery_rate"] = recovered_faults / total_faults
    else:
        aggregate["detection_rate"] = 1.0
        aggregate["recovery_rate"] = 1.0
    
    # Calculate MTTD and MTTR
    all_detection_times = []
    all_recovery_times = []
    
    for metrics in episode_metrics:
        all_detection_times.extend(metrics["fault_detection_times"])
        all_recovery_times.extend(metrics["fault_recovery_times"])
    
    if all_detection_times:
        aggregate["mttd"] = np.mean(all_detection_times)
    else:
        aggregate["mttd"] = 0
    
    if all_recovery_times:
        aggregate["mttr"] = np.mean(all_recovery_times)
    else:
        aggregate["mttr"] = 0
    
    # Calculate action distribution
    all_actions = []
    for m in episode_metrics:
        all_actions.extend(m["actions"])
    
    unique_actions, counts = np.unique(all_actions, return_counts=True)
    for action, count in zip(unique_actions, counts):
        aggregate["action_distribution"][int(action)] = int(count)
    
    # Calculate decision source distribution
    all_sources = []
    for m in episode_metrics:
        if "decision_sources" in m:
            all_sources.extend(m["decision_sources"])
    
    if all_sources:
        unique_sources, counts = np.unique(all_sources, return_counts=True)
        for source, count in zip(unique_sources, counts):
            aggregate["decision_source_distribution"][source] = int(count)
    
    # Calculate SFRI score
    sfri_scores = []
    for metrics in episode_metrics:
        sfri_scores.append(calculate_sfri(metrics, max_steps))
    
    aggregate["sfri"] = np.mean(sfri_scores)
    
    return aggregate

def main():
    """Main function to compare the original and enhanced hybrid agents."""
    parser = argparse.ArgumentParser(description="Compare original and enhanced hybrid FDIR agents")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="results/hybrid_comparison.json", help="Output file")
    args = parser.parse_args()
    
    # Environment configuration
    env_config = {
        "fault_probability": 0.02,
        "fault_sample_mode": "random"
    }
    
    # Create agents
    original_agent = MockHybridFDIRAgent()
    enhanced_agent = MockEnhancedHybridFDIRAgent()
    
    # Evaluate agents
    print("Evaluating original hybrid agent...")
    original_results = evaluate_agent(
        env_config,
        original_agent,
        num_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed
    )
    
    print("Evaluating enhanced hybrid agent...")
    enhanced_results = evaluate_agent(
        env_config,
        enhanced_agent,
        num_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed
    )
    
    # Combine results
    results = {
        "original_hybrid": {
            "aggregate_metrics": original_results["aggregate_metrics"]
        },
        "enhanced_hybrid": {
            "aggregate_metrics": enhanced_results["aggregate_metrics"],
            "false_positives_prevented": enhanced_agent.false_positive_prevented
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Print summary comparison
    print("\nPerformance Comparison:")
    print("-" * 60)
    print(f"Metric                   | Original Hybrid | Enhanced Hybrid")
    print("-" * 60)
    print(f"Mean Reward              | {original_results['aggregate_metrics']['mean_reward']:14.2f} | {enhanced_results['aggregate_metrics']['mean_reward']:14.2f}")
    print(f"False Positives          | {original_results['aggregate_metrics']['total_false_positives']:14d} | {enhanced_results['aggregate_metrics']['total_false_positives']:14d}")
    print(f"False Positive Rate      | {original_results['aggregate_metrics']['false_positive_rate']:14.2f} | {enhanced_results['aggregate_metrics']['false_positive_rate']:14.2f}")
    print(f"Detection Rate           | {original_results['aggregate_metrics']['detection_rate']*100:13.1f}% | {enhanced_results['aggregate_metrics']['detection_rate']*100:13.1f}%")
    print(f"Recovery Rate            | {original_results['aggregate_metrics']['recovery_rate']*100:13.1f}% | {enhanced_results['aggregate_metrics']['recovery_rate']*100:13.1f}%")
    print(f"Mean Time To Detect      | {original_results['aggregate_metrics']['mttd']:14.1f} | {enhanced_results['aggregate_metrics']['mttd']:14.1f}")
    print(f"Mean Time To Recover     | {original_results['aggregate_metrics']['mttr']:14.1f} | {enhanced_results['aggregate_metrics']['mttr']:14.1f}")
    print(f"SFRI Score               | {original_results['aggregate_metrics']['sfri']:14.1f} | {enhanced_results['aggregate_metrics']['sfri']:14.1f}")
    
    # Calculate false positive reduction
    if original_results['aggregate_metrics']['total_false_positives'] > 0:
        reduction = 1.0 - (enhanced_results['aggregate_metrics']['total_false_positives'] / 
                         original_results['aggregate_metrics']['total_false_positives'])
        print(f"False Positive Reduction | {'-':14s} | {reduction*100:13.1f}%")
    
    print(f"False Positives Prevented| {'-':14s} | {enhanced_agent.false_positive_prevented:14d}")
    
    print("-" * 60)

if __name__ == "__main__":
    main() 
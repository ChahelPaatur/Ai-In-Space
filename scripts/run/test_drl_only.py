#!/usr/bin/env python3
"""
DRL Agent Isolation Test Script

This script tests ONLY the DRL agent to debug why it's still detecting faults
despite having "impossible" emergency detection thresholds.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.drl_agent import PPOAgent
from src.environment import SpacecraftEnv

def test_drl_agent():
    """Test DRL agent in isolation to see what's happening."""
    
    print("=== DRL AGENT ISOLATION TEST ===")
    print("Testing DRL agent with 'impossible' thresholds to see why it still detects faults\n")
    
    # Initialize environment and agent
    env = SpacecraftEnv()
    agent = PPOAgent(obs_size=15, action_size=env.action_space.n)
    
    # Load the trained model if available
    try:
        agent.load_model("ppo_agent.pth")
        print("✅ Loaded trained DRL model")
    except:
        print("⚠️ No trained model found - using random agent")
    
    # Test configuration
    num_episodes = 10
    detailed_logging = True
    
    print(f"\n🔍 Running {num_episodes} test episodes with detailed logging...")
    print("Looking for:")
    print("  - Emergency detection triggers")
    print("  - Action selection patterns") 
    print("  - Confidence levels")
    print("  - Actual vs expected behavior\n")
    
    all_actions = []
    all_rewards = []
    detection_events = []
    action_breakdown = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        observation, info = env.reset()
        total_reward = 0
        episode_actions = []
        step_count = 0
        faults_detected_this_episode = 0
        
        for step in range(200):  # Max 200 steps per episode
            # Get action with detailed diagnostics
            action, log_prob, value, diagnostics = agent.enhanced_get_action(observation, info)
            
            # Log detailed information
            if detailed_logging and step < 20:  # First 20 steps for detail
                print(f"  Step {step:3d}: Action={action}, Source={diagnostics.get('source', 'unknown')}")
                if diagnostics.get('emergency_action') is not None:
                    print(f"           🚨 EMERGENCY ACTION DETECTED: {diagnostics['emergency_action']}")
                    detection_events.append({
                        'episode': episode,
                        'step': step,
                        'action': action,
                        'source': diagnostics.get('source'),
                        'observation': observation.copy()
                    })
                    faults_detected_this_episode += 1
                
                # Check observation values that should trigger emergency detection
                eps_voltage = observation[2] if len(observation) > 2 else "N/A"
                thermal_temp = observation[6] if len(observation) > 6 else "N/A"
                attitude_error = observation[9] if len(observation) > 9 else "N/A"
                
                print(f"           Obs: EPS={eps_voltage:.3f}, Thermal={thermal_temp:.3f}, Attitude={attitude_error:.3f}")
                
                # Check if these values should trigger emergency detection
                should_trigger_eps = eps_voltage < -1.5 if eps_voltage != "N/A" else False
                should_trigger_thermal = thermal_temp > 1.5 if thermal_temp != "N/A" else False
                should_trigger_attitude = abs(attitude_error) > 1.5 if attitude_error != "N/A" else False
                
                if should_trigger_eps or should_trigger_thermal or should_trigger_attitude:
                    print(f"           ⚠️ Should trigger emergency: EPS={should_trigger_eps}, Thermal={should_trigger_thermal}, Attitude={should_trigger_attitude}")
            
            # Track actions
            episode_actions.append(action)
            action_breakdown[action] += 1
            
            # Take step in environment
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        all_actions.extend(episode_actions)
        all_rewards.append(total_reward)
        
        print(f"  Episode Summary:")
        print(f"    Total Reward: {total_reward:.2f}")
        print(f"    Steps: {step_count}")
        print(f"    Faults Detected: {faults_detected_this_episode}")
        print(f"    Actions: {set(episode_actions)}")
    
    # Final Analysis
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Episodes run: {num_episodes}")
    print(f"Total detection events: {len(detection_events)}")
    print(f"Average reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    
    print(f"\n📊 Action Breakdown:")
    total_actions = sum(action_breakdown.values())
    for action, count in action_breakdown.items():
        percentage = (count / total_actions * 100) if total_actions > 0 else 0
        action_name = {0: "NO_OP", 1: "RECOVER_EPS", 2: "RECOVER_ADCS", 3: "RECOVER_TCS", 
                      4: "HEATER_ON", 5: "HEATER_OFF", 6: "RESET_GYRO", 7: "SAFE_MODE", 8: "NOMINAL_MODE"}
        print(f"  {action} ({action_name.get(action, 'UNKNOWN')}): {count:4d} ({percentage:5.1f}%)")
    
    print(f"\n🔍 Detection Events Analysis:")
    if detection_events:
        print(f"  Total emergency detections: {len(detection_events)}")
        for i, event in enumerate(detection_events[:5]):  # Show first 5
            print(f"  Event {i+1}: Episode {event['episode']}, Step {event['step']}, Action {event['action']}, Source: {event['source']}")
            obs = event['observation']
            print(f"    Observation: EPS={obs[2]:.3f}, Thermal={obs[6]:.3f}, Attitude={obs[9]:.3f}")
    else:
        print("  ✅ No emergency detections found (as expected with impossible thresholds)")
    
    # Check if DRL is using its neural network vs fallback logic
    print(f"\n🧠 Neural Network vs Fallback Analysis:")
    fallback_actions = action_breakdown[0]  # NO_OP actions
    recovery_actions = sum([action_breakdown[i] for i in [1, 2, 3]])  # Recovery actions
    other_actions = total_actions - fallback_actions - recovery_actions
    
    print(f"  NO_OP actions (likely fallback): {fallback_actions} ({fallback_actions/total_actions*100:.1f}%)")
    print(f"  Recovery actions: {recovery_actions} ({recovery_actions/total_actions*100:.1f}%)")
    print(f"  Other actions: {other_actions} ({other_actions/total_actions*100:.1f}%)")
    
    if recovery_actions > 0:
        print(f"  ⚠️ DRL is still taking recovery actions despite impossible thresholds!")
        print(f"     This suggests the neural network is overriding emergency detection logic.")
    else:
        print(f"  ✅ DRL is not taking recovery actions (emergency detection working)")
    
    print(f"\n=== CONCLUSION ===")
    if recovery_actions > 0:
        print("🚨 ISSUE FOUND: DRL is still detecting/recovering from faults")
        print("   The neural network is likely making decisions independent of emergency detection")
        print("   We need to make the neural network itself more conservative")
    else:
        print("✅ DRL emergency detection is properly disabled")
        print("   Any remaining detection might be through the neural network path")

if __name__ == "__main__":
    test_drl_agent() 
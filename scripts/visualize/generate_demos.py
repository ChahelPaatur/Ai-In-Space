import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import sys
import torch
import json
import time

# Add the project root to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create output directory if it doesn't exist
os.makedirs("static/plots/demos", exist_ok=True)

def generate_dashboard_screenshot():
    """Create a static mockup of the dashboard interface"""
    print("Generating dashboard screenshot...")
    
    # Create a mockup of the dashboard layout
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    # Add title
    fig.suptitle("Spacecraft FDIR Agent Dashboard", fontsize=16, y=0.98)
    
    # Agent performance comparison graph
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_title("Agent Performance Comparison")
    agents = ['Rule-based', 'DRL', 'Hybrid']
    metrics = {'MTTD': [36.7, 21.8, 18.2], 
               'MTTR': [147.8, 152.0, 146.9], 
               'SFRI': [38.5, 37.9, 40.0]}
    x = np.arange(len(agents))
    width = 0.25
    ax1.bar(x - width, metrics['MTTD'], width, label='MTTD')
    ax1.bar(x, metrics['MTTR'], width, label='MTTR')
    ax1.bar(x + width, metrics['SFRI'], width, label='SFRI')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subsystem status indicators
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Subsystem Status")
    subsystems = ['EPS', 'ADCS', 'TCS', 'Comms', 'Payload']
    status = [0.8, 0.9, 0.7, 1.0, 0.95]  # Health values from 0 to 1
    colors = ['green' if s > 0.8 else 'yellow' if s > 0.6 else 'red' for s in status]
    ax2.barh(subsystems, status, color=colors)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Health")
    for i, v in enumerate(status):
        ax2.text(v + 0.05, i, f"{v:.1f}", va='center')
    
    # Time series of temperature
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Temperature")
    time = np.arange(100)
    temp = 20 + 5 * np.sin(time / 10) + np.random.normal(0, 0.5, 100)
    # Add a simulated fault
    temp[50:70] += np.linspace(0, 8, 20)
    temp[70:] += 8 - np.linspace(0, 8, 30)
    ax3.plot(time, temp, 'b-')
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("°C")
    ax3.grid(True, alpha=0.3)
    ax3.axvspan(50, 70, alpha=0.2, color='red')
    
    # Time series of battery
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title("Battery State of Charge")
    soc = 0.8 - 0.2 * np.sin(time / 30) + np.random.normal(0, 0.02, 100)
    ax4.plot(time, soc, 'g-')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("SOC")
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Time series of attitude error
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_title("Attitude Error")
    error = 0.05 * np.exp(-0.1 * time) + 0.01 * np.sin(time) + np.random.normal(0, 0.005, 100)
    # Add a simulated correction
    error[60:] = 0.03 * np.exp(-0.2 * (time[60:] - 60)) + np.random.normal(0, 0.002, 40)
    ax5.plot(time, error, 'r-')
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Error (rad)")
    ax5.grid(True, alpha=0.3)
    
    # Agent decision log
    ax6 = fig.add_subplot(gs[2, :])
    ax6.set_title("Agent Decision Log")
    # Create a table with some mock data
    data = [
        ['90', 'Hybrid', 'Temperature Rising', 'HeaterOFF', 'High', 'Success'],
        ['85', 'DRL', 'Battery Drain', 'EnterSafe', 'Medium', 'Success'],
        ['70', 'Rule-based', 'Attitude Drift', 'RecoverADCS', 'N/A', 'Partial'],
        ['65', 'Hybrid', 'Comms Failure', 'RecoverComms', 'High', 'Success'],
        ['50', 'DRL', 'Reaction Wheel Fault', 'ResetGyroBias', 'Low', 'Failure'],
    ]
    column_labels = ['Time (s)', 'Agent', 'Detected Issue', 'Action', 'Confidence', 'Outcome']
    ax6.axis('tight')
    ax6.axis('off')
    table = ax6.table(cellText=data, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Add control panel mockup
    plt.figtext(0.02, 0.02, "Control Panel", fontsize=10, weight='bold')
    plt.figtext(0.02, 0.01, "Active Agent: Hybrid | Mode: Auto | Telemetry Rate: 10Hz", fontsize=8)
    
    # Save the dashboard mockup
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("static/plots/demos/dashboard_demo.png", dpi=120)
    plt.close(fig)
    print("Saved dashboard screenshot to static/plots/demos/dashboard_demo.png")

def generate_agent_comparison():
    """Create static images showing comparisons between agents for different faults."""
    print("Generating agent comparison visualizations...")
    
    # Create comparison for temperature response
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Agent Response Comparison: Thermal Fault", fontsize=16)
    
    # Create simulated temperature data for each agent
    steps = np.arange(100, dtype=np.float64)
    fault_step = 30
    
    # Rule-based temperature response (slower detection, stable recovery)
    temp_rule = 20 + np.zeros_like(steps, dtype=np.float64)
    temp_rule[fault_step:] = 20 - 0.5 * (steps[fault_step:] - fault_step)
    temp_rule[fault_step+40:] = 5 + 0.5 * (steps[fault_step+40:] - (fault_step+40))
    temp_rule[fault_step+60:] = 15  # Stabilize
    
    # DRL temperature response (faster detection, oscillatory)
    temp_drl = 20 + np.zeros_like(steps, dtype=np.float64)
    temp_drl[fault_step:] = 20 - 0.8 * (steps[fault_step:] - fault_step) 
    temp_drl[fault_step+15:] = 8 + 1.0 * (steps[fault_step+15:] - (fault_step+15))
    temp_drl[fault_step+25:] = 18 - 0.3 * (steps[fault_step+25:] - (fault_step+25))
    temp_drl[fault_step+35:] = 15 + 0.3 * (steps[fault_step+35:] - (fault_step+35))
    temp_drl[fault_step+45:] = 18 - 0.1 * (steps[fault_step+45:] - (fault_step+45))
    temp_drl[fault_step+60:] = 16  # Stabilize
    
    # Hybrid temperature response (fast detection, stable recovery)
    temp_hybrid = 20 + np.zeros_like(steps, dtype=np.float64)
    temp_hybrid[fault_step:] = 20 - 0.7 * (steps[fault_step:] - fault_step)
    temp_hybrid[fault_step+20:] = 6 + 0.7 * (steps[fault_step+20:] - (fault_step+20))
    temp_hybrid[fault_step+30:] = 16 - 0.05 * (steps[fault_step+30:] - (fault_step+30))
    temp_hybrid[fault_step+60:] = 15  # Stabilize
    
    # Add some noise
    temp_rule += np.random.normal(0, 0.2, len(steps))
    temp_drl += np.random.normal(0, 0.3, len(steps))
    temp_hybrid += np.random.normal(0, 0.25, len(steps))
    
    plt.plot(steps, temp_rule, 'b-', label='Rule-based')
    plt.plot(steps, temp_drl, 'r-', label='DRL')
    plt.plot(steps, temp_hybrid, 'g-', label='Hybrid')
    plt.axvline(x=fault_step, color='k', linestyle='--', label='Fault Injected')
    
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("static/plots/demos/thermal_response_comparison.png", dpi=120)
    plt.close(fig)
    print("Saved thermal response comparison to static/plots/demos/thermal_response_comparison.png")
    
    # Create comparison for battery SOC response
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Agent Response Comparison: Power Fault", fontsize=16)
    
    # Create simulated SOC data for each agent
    fault_step = 30
    
    # Rule-based SOC response
    soc_rule = np.zeros_like(steps, dtype=np.float64)
    soc_rule[:] = 0.8 - 0.005 * steps
    soc_rule[fault_step:] = soc_rule[fault_step] - 0.01 * (steps[fault_step:] - fault_step)
    soc_rule[fault_step+30:] = soc_rule[fault_step+30] - 0.001 * (steps[fault_step+30:] - (fault_step+30))
    
    # DRL SOC response
    soc_drl = np.zeros_like(steps, dtype=np.float64)
    soc_drl[:] = 0.8 - 0.005 * steps
    soc_drl[fault_step:] = soc_drl[fault_step] - 0.01 * (steps[fault_step:] - fault_step)
    soc_drl[fault_step+10:] = soc_drl[fault_step+10] - 0.001 * (steps[fault_step+10:] - (fault_step+10))
    
    # Hybrid SOC response
    soc_hybrid = np.zeros_like(steps, dtype=np.float64)
    soc_hybrid[:] = 0.8 - 0.005 * steps
    soc_hybrid[fault_step:] = soc_hybrid[fault_step] - 0.01 * (steps[fault_step:] - fault_step)
    soc_hybrid[fault_step+15:] = soc_hybrid[fault_step+15] - 0.001 * (steps[fault_step+15:] - (fault_step+15))
    
    # Add some noise
    soc_rule += np.random.normal(0, 0.005, len(steps))
    soc_drl += np.random.normal(0, 0.005, len(steps))
    soc_hybrid += np.random.normal(0, 0.005, len(steps))
    
    plt.plot(steps, soc_rule, 'b-', label='Rule-based')
    plt.plot(steps, soc_drl, 'r-', label='DRL')
    plt.plot(steps, soc_hybrid, 'g-', label='Hybrid')
    plt.axvline(x=fault_step, color='k', linestyle='--', label='Fault Injected')
    
    plt.xlabel('Time Step')
    plt.ylabel('Battery State of Charge')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("static/plots/demos/battery_response_comparison.png", dpi=120)
    plt.close(fig)
    print("Saved battery response comparison to static/plots/demos/battery_response_comparison.png")
    
    # Create comparison for ADCS response
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Agent Response Comparison: Attitude Control Fault", fontsize=16)
    
    # Create simulated attitude error data for each agent
    fault_step = 30
    
    # Rule-based attitude error response
    att_err_rule = np.zeros_like(steps, dtype=np.float64)
    att_err_rule[:] = 0.01 + 0.005 * np.sin(steps / 5)
    att_err_rule[fault_step:] = 0.01 + 0.02 * (steps[fault_step:] - fault_step)
    att_err_rule[fault_step+40:] = att_err_rule[fault_step+40-1] * np.exp(-0.1 * (steps[fault_step+40:] - (fault_step+40)))
    
    # DRL attitude error response
    att_err_drl = np.zeros_like(steps, dtype=np.float64)
    att_err_drl[:] = 0.01 + 0.005 * np.sin(steps / 5)
    att_err_drl[fault_step:] = 0.01 + 0.02 * (steps[fault_step:] - fault_step)
    att_err_drl[fault_step+10:] = att_err_drl[fault_step+10-1] * np.exp(-0.15 * (steps[fault_step+10:] - (fault_step+10)))
    
    # Hybrid attitude error response
    att_err_hybrid = np.zeros_like(steps, dtype=np.float64)
    att_err_hybrid[:] = 0.01 + 0.005 * np.sin(steps / 5)
    att_err_hybrid[fault_step:] = 0.01 + 0.02 * (steps[fault_step:] - fault_step)
    att_err_hybrid[fault_step+15:] = att_err_hybrid[fault_step+15-1] * np.exp(-0.2 * (steps[fault_step+15:] - (fault_step+15)))
    
    # Add some noise
    att_err_rule += np.random.normal(0, 0.002, len(steps))
    att_err_drl += np.random.normal(0, 0.003, len(steps))
    att_err_hybrid += np.random.normal(0, 0.0025, len(steps))
    
    plt.plot(steps, att_err_rule, 'b-', label='Rule-based')
    plt.plot(steps, att_err_drl, 'r-', label='DRL')
    plt.plot(steps, att_err_hybrid, 'g-', label='Hybrid')
    plt.axvline(x=fault_step, color='k', linestyle='--', label='Fault Injected')
    
    plt.xlabel('Time Step')
    plt.ylabel('Attitude Error (rad)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("static/plots/demos/attitude_response_comparison.png", dpi=120)
    plt.close(fig)
    print("Saved attitude response comparison to static/plots/demos/attitude_response_comparison.png")

if __name__ == "__main__":
    # Generate the dashboard screenshot
    generate_dashboard_screenshot()
    
    # Generate agent comparison visualizations
    generate_agent_comparison()
    
    print("All demonstration assets have been generated!") 
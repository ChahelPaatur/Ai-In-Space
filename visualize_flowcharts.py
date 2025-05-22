import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

# --- Configuration ---
OUTPUT_DIR = "static/plots"

def create_rule_based_flowchart():
    """
    Create a flowchart visualizing the rule-based FDIR decision logic.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 15))
    
    # Define colors for different node types
    colors = {
        'start': '#AED6F1',  # Light blue
        'decision': '#FADBD8',  # Light red/pink
        'action': '#ABEBC6',  # Light green
        'end': '#D5D8DC'  # Light gray
    }
    
    # Starting point
    ax.add_patch(Rectangle((2, 12), 4, 1, facecolor=colors['start'], edgecolor='black', alpha=0.7))
    ax.text(4, 12.5, 'Start: Get Telemetry', ha='center', va='center', fontsize=12)
    
    # Mode Management Decision
    ax.add_patch(Rectangle((2, 10), 4, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 10.5, 'SoC < Critical Threshold?', ha='center', va='center', fontsize=12)
    
    # Arrow from Start to Mode Management
    ax.add_patch(FancyArrowPatch((4, 12), (4, 11), arrowstyle='->', connectionstyle='arc3', color='black'))
    
    # Enter Safe Mode Action
    ax.add_patch(Rectangle((7, 10), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(9, 10.5, 'Enter Safe Mode', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((6, 10.5), (7, 10.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(6.5, 10.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # EPS Check
    ax.add_patch(Rectangle((2, 8), 4, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 8.5, 'Bus Voltage < Threshold?', ha='center', va='center', fontsize=12)
    
    # Arrow from SoC to EPS
    ax.add_patch(FancyArrowPatch((4, 10), (4, 9), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, 9.5, 'No', ha='center', va='center', fontsize=10)
    
    # Recover EPS Action
    ax.add_patch(Rectangle((7, 8), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(9, 8.5, 'Recover EPS', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((6, 8.5), (7, 8.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(6.5, 8.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # ADCS Check - Tumbling
    ax.add_patch(Rectangle((2, 6), 4, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 6.5, 'Angular Velocity > Threshold\nor Attitude Error > Threshold?', ha='center', va='center', fontsize=11)
    
    # Arrow from EPS to ADCS
    ax.add_patch(FancyArrowPatch((4, 8), (4, 7), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, 7.5, 'No', ha='center', va='center', fontsize=10)
    
    # Recover ADCS Action
    ax.add_patch(Rectangle((7, 6), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(9, 6.5, 'Recover ADCS', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((6, 6.5), (7, 6.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(6.5, 6.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # ADCS Gyro Check
    ax.add_patch(Rectangle((1, 4), 6, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 4.5, 'Moderate Attitude Error and Not Tumbling?', ha='center', va='center', fontsize=11)
    
    # Arrow from ADCS to Gyro Check
    ax.add_patch(FancyArrowPatch((4, 6), (4, 5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, 5.5, 'No', ha='center', va='center', fontsize=10)
    
    # Reset Gyro Bias Action
    ax.add_patch(Rectangle((8, 4), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(10, 4.5, 'Reset Gyro Bias', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((7, 4.5), (8, 4.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(7.5, 4.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # TCS Critical Check
    ax.add_patch(Rectangle((2, 2), 4, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 2.5, 'Temperature Outside\nCritical Bounds?', ha='center', va='center', fontsize=11)
    
    # Arrow from Gyro to TCS
    ax.add_patch(FancyArrowPatch((4, 4), (4, 3), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, 3.5, 'No', ha='center', va='center', fontsize=10)
    
    # Recover TCS Action
    ax.add_patch(Rectangle((7, 2), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(9, 2.5, 'Recover TCS', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((6, 2.5), (7, 2.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(6.5, 2.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # TCS Heater Control
    ax.add_patch(Rectangle((1, 0), 6, 1, facecolor=colors['decision'], edgecolor='black', alpha=0.7))
    ax.text(4, 0.5, 'Temperature Outside Normal Bounds?', ha='center', va='center', fontsize=11)
    
    # Arrow from TCS Critical to Heater
    ax.add_patch(FancyArrowPatch((4, 2), (4, 1), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, 1.5, 'No', ha='center', va='center', fontsize=10)
    
    # Heater Action
    ax.add_patch(Rectangle((8, 0), 4, 1, facecolor=colors['action'], edgecolor='black', alpha=0.7))
    ax.text(10, 0.5, 'Heater ON/OFF', ha='center', va='center', fontsize=12)
    
    # Arrow for Yes path
    ax.add_patch(FancyArrowPatch((7, 0.5), (8, 0.5), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(7.5, 0.7, 'Yes', ha='center', va='center', fontsize=10)
    
    # Default No-op
    ax.add_patch(Rectangle((2, -2), 4, 1, facecolor=colors['end'], edgecolor='black', alpha=0.7))
    ax.text(4, -1.5, 'No-op (Default)', ha='center', va='center', fontsize=12)
    
    # Arrow from Heater to No-op
    ax.add_patch(FancyArrowPatch((4, 0), (4, -1), arrowstyle='->', connectionstyle='arc3', color='black'))
    ax.text(3.7, -0.5, 'No', ha='center', va='center', fontsize=10)
    
    # Set up the plot
    ax.set_xlim(-1, 14)
    ax.set_ylim(-3, 14)
    ax.set_title('Rule-Based FDIR Logic Flowchart', fontsize=16)
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rule_based_flowchart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Rule-Based FDIR flowchart to {os.path.join(OUTPUT_DIR, 'rule_based_flowchart.png')}")

def create_sfri_diagram():
    """
    Create a diagram explaining the new SFRI (Stability-Integrated Fault Recovery Index) metric.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Component boxes
    components = [
        ('Fault Detection Accuracy', 0.5, 0.85, '#AED6F1'),
        ('Recovery Time (MTTR)', 0.5, 0.65, '#FADBD8'),
        ('System Stability Impact', 0.5, 0.45, '#ABEBC6'),
        ('False Positive Penalty', 0.5, 0.25, '#F9E79F'),
        ('SFRI Score', 0.5, 0.05, '#D7BDE2')
    ]
    
    # Draw components
    for name, x, y, color in components:
        ax.add_patch(Rectangle((x-0.3, y-0.07), 0.6, 0.14, facecolor=color, edgecolor='black', alpha=0.7))
        ax.text(x, y, name, ha='center', va='center', fontsize=12)
    
    # Draw arrows connecting components
    for i in range(len(components) - 1):
        ax.add_patch(FancyArrowPatch(
            (components[i][1], components[i][2] - 0.07),
            (components[i+1][1], components[i+1][2] + 0.07),
            arrowstyle='->',
            connectionstyle='arc3',
            color='black'
        ))
    
    # Add formula
    formula = "SFRI = (α × Detection Accuracy) - (β × MTTR) - (γ × Stability Impact) - (δ × False Positives)"
    ax.text(0.5, -0.15, formula, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add explanations
    explanations = [
        "Detection Accuracy: % of faults correctly identified",
        "MTTR: Mean Time To Recovery (lower is better)",
        "Stability Impact: Penalty for destabilizing other subsystems",
        "False Positives: Penalty for incorrect recovery actions"
    ]
    
    for i, exp in enumerate(explanations):
        ax.text(1.1, 0.8 - i*0.2, exp, ha='left', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Set up the plot
    ax.set_xlim(0, 2)
    ax.set_ylim(-0.2, 1)
    ax.set_title('Stability-Integrated Fault Recovery Index (SFRI)', fontsize=16)
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sfri_metric.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved SFRI metric diagram to {os.path.join(OUTPUT_DIR, 'sfri_metric.png')}")

def main():
    """Main function to generate all flowcharts."""
    print("--- Generating Flowcharts ---")
    
    # Create rule-based FDIR flowchart
    create_rule_based_flowchart()
    
    # Create SFRI metric diagram
    create_sfri_diagram()
    
    print("--- Flowchart Generation Complete ---")

if __name__ == "__main__":
    main() 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as PathEffects
from matplotlib.path import Path
import matplotlib.patches as patches

# Output directory
OUTPUT_DIR = 'static/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def create_drl_architecture():
    """
    Create Figure 8a: DRL Agent Architecture diagram showing the actor-critic network structure.
    
    # Paper reference: Section 3.3 "DRL Agent (PPOAgent)" - This diagram illustrates the
    # neural network architecture with shared representation layers and separate policy (actor)
    # and value (critic) heads as described in the paper.
    """
    print("Generating DRL architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and styles
    node_color = '#3498db'  # Blue
    actor_color = '#e74c3c'  # Red
    critic_color = '#2ecc71'  # Green
    arrow_color = '#7f8c8d'  # Gray
    
    # Layer positions
    input_y = 0.5
    hidden1_y = 2.0
    hidden2_y = 3.5
    output_y = 5.0
    
    # Draw input layer (observation)
    input_nodes = 8
    input_x = np.linspace(2, 8, input_nodes)
    for i, x in enumerate(input_x):
        circle = plt.Circle((x, input_y), 0.2, color=node_color, alpha=0.7)
        ax.add_patch(circle)
    
    # Draw hidden layers
    hidden_nodes = [64, 64]  # Hidden layer 1 and 2
    hidden1_x = np.linspace(2, 8, 5)  # Show 5 representative nodes
    hidden2_x = np.linspace(2, 8, 5)
    
    # Draw first hidden layer
    for x in hidden1_x:
        circle = plt.Circle((x, hidden1_y), 0.2, color=node_color, alpha=0.7)
        ax.add_patch(circle)
    
    # Draw second hidden layer
    for x in hidden2_x:
        circle = plt.Circle((x, hidden2_y), 0.2, color=node_color, alpha=0.7)
        ax.add_patch(circle)
    
    # Draw output layers
    actor_x = 3.5
    critic_x = 6.5
    
    # Actor head (Policy)
    circle = plt.Circle((actor_x, output_y), 0.3, color=actor_color, alpha=0.7)
    ax.add_patch(circle)
    
    # Critic head (Value)
    circle = plt.Circle((critic_x, output_y), 0.3, color=critic_color, alpha=0.7)
    ax.add_patch(circle)
    
    # Draw connections between layers (simplified)
    # Input to Hidden 1
    for x_in in input_x[::2]:  # Draw fewer lines for clarity
        for x_h1 in hidden1_x:
            ax.plot([x_in, x_h1], [input_y, hidden1_y], color=arrow_color, linewidth=0.5, alpha=0.3)
    
    # Hidden 1 to Hidden 2
    for x_h1 in hidden1_x:
        for x_h2 in hidden2_x:
            ax.plot([x_h1, x_h2], [hidden1_y, hidden2_y], color=arrow_color, linewidth=0.5, alpha=0.3)
    
    # Hidden 2 to Actor
    for x_h2 in hidden2_x:
        ax.plot([x_h2, actor_x], [hidden2_y, output_y], color=actor_color, linewidth=0.5, alpha=0.4)
    
    # Hidden 2 to Critic
    for x_h2 in hidden2_x:
        ax.plot([x_h2, critic_x], [hidden2_y, output_y], color=critic_color, linewidth=0.5, alpha=0.4)
    
    # Add labels
    ax.text(1.0, input_y, "Input\nLayer", fontsize=12, ha='right', va='center')
    ax.text(1.0, hidden1_y, "Hidden\nLayer 1", fontsize=12, ha='right', va='center')
    ax.text(1.0, hidden2_y, "Hidden\nLayer 2", fontsize=12, ha='right', va='center')
    ax.text(actor_x, output_y + 0.5, "Policy Head\n(Actor)", fontsize=12, ha='center', va='center', color=actor_color)
    ax.text(critic_x, output_y + 0.5, "Value Head\n(Critic)", fontsize=12, ha='center', va='center', color=critic_color)
    
    # Add size information
    ax.text(9.0, input_y, f"(size: {input_nodes})", fontsize=10, ha='left', va='center')
    ax.text(9.0, hidden1_y, f"(size: {hidden_nodes[0]})", fontsize=10, ha='left', va='center')
    ax.text(9.0, hidden2_y, f"(size: {hidden_nodes[1]})", fontsize=10, ha='left', va='center')
    ax.text(actor_x, output_y - 0.6, "(size: 9)", fontsize=10, ha='center', va='center')
    ax.text(critic_x, output_y - 0.6, "(size: 1)", fontsize=10, ha='center', va='center')
    
    # Add ellipses to represent more nodes
    ax.text(5, input_y, "...", fontsize=16, ha='center', va='center')
    ax.text(5, hidden1_y, "...", fontsize=16, ha='center', va='center')
    ax.text(5, hidden2_y, "...", fontsize=16, ha='center', va='center')
    
    # Add layer descriptions
    ax.text(10, input_y, "Normalized observation", fontsize=10, ha='left', va='center')
    ax.text(10, hidden1_y, "ReLU Activation", fontsize=10, ha='left', va='center')
    ax.text(10, hidden2_y, "ReLU Activation", fontsize=10, ha='left', va='center')
    ax.text(actor_x, output_y - 1.0, "Softmax (Action Probabilities)", fontsize=10, ha='center', va='center')
    ax.text(critic_x, output_y - 1.0, "Linear (Value Estimate)", fontsize=10, ha='center', va='center')
    
    # Add DRL agent label at the top
    plt.text(5, 6.0, "DRL Agent Architecture", fontsize=16, ha='center', va='bottom', weight='bold')
    
    # Configure the axes
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'drl_architecture.png'))
    plt.close()
    print("  DRL architecture diagram saved to", os.path.join(OUTPUT_DIR, 'drl_architecture.png'))

def create_hybrid_architecture():
    """
    Create Figure 8b: Hybrid Agent Architecture showing the confidence-based arbitration mechanism.
    
    # Paper reference: Section 3.4 "Hybrid Agent (HybridFDIRAgent)" - This diagram illustrates
    # the arbitration mechanism that determines whether the rule-based or DRL component makes
    # the final decision based on confidence levels and safety considerations.
    """
    print("Generating hybrid architecture diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    rule_color = '#3498db'  # Blue
    drl_color = '#e74c3c'   # Red
    hybrid_color = '#9b59b6'  # Purple
    box_color = '#ecf0f1'   # Light gray
    
    # Draw the main boxes
    rule_box = Rectangle((1, 1), 3, 2, facecolor=box_color, edgecolor='black', alpha=0.7)
    drl_box = Rectangle((8, 1), 3, 2, facecolor=box_color, edgecolor='black', alpha=0.7)
    arbiter_box = Rectangle((4.5, 4), 3, 2, facecolor=box_color, edgecolor='black', alpha=0.7, linestyle='-')
    output_box = Rectangle((4.5, 7), 3, 1, facecolor=box_color, edgecolor='black', alpha=0.7)
    
    ax.add_patch(rule_box)
    ax.add_patch(drl_box)
    ax.add_patch(arbiter_box)
    ax.add_patch(output_box)
    
    # Add labels
    ax.text(2.5, 2, "Rule-Based\nComponent", ha='center', va='center', fontsize=12, color=rule_color, weight='bold')
    ax.text(9.5, 2, "DRL\nComponent", ha='center', va='center', fontsize=12, color=drl_color, weight='bold')
    ax.text(6, 5, "Confidence-Based\nArbitration", ha='center', va='center', fontsize=12, color='black', weight='bold')
    ax.text(6, 7.5, "Final Action", ha='center', va='center', fontsize=12, weight='bold', color=hybrid_color)
    
    # Draw arrows
    # Input arrow to rule-based
    input_arrow1 = FancyArrowPatch((2.5, 0), (2.5, 1), arrowstyle='->', mutation_scale=20, color=rule_color, linewidth=2)
    ax.add_patch(input_arrow1)
    
    # Input arrow to DRL
    input_arrow2 = FancyArrowPatch((9.5, 0), (9.5, 1), arrowstyle='->', mutation_scale=20, color=drl_color, linewidth=2)
    ax.add_patch(input_arrow2)
    
    # Rule-based to arbitration
    rule_arrow = FancyArrowPatch((2.5, 3), (5.0, 4), arrowstyle='->', mutation_scale=20, color=rule_color, linewidth=2)
    ax.add_patch(rule_arrow)
    
    # DRL to arbitration
    drl_arrow = FancyArrowPatch((9.5, 3), (7.0, 4), arrowstyle='->', mutation_scale=20, color=drl_color, linewidth=2)
    ax.add_patch(drl_arrow)
    
    # Arbitration to output
    output_arrow = FancyArrowPatch((6, 6), (6, 7), arrowstyle='->', mutation_scale=20, color=hybrid_color, linewidth=2)
    ax.add_patch(output_arrow)
    
    # Add input label
    ax.text(2.5, -0.2, "Telemetry & System Status", ha='center', va='top', fontsize=10)
    ax.text(9.5, -0.2, "Normalized Observation", ha='center', va='top', fontsize=10)
    
    # Add decision criteria text
    decision_text = (
        "Decision Logic:\n"
        "1. Safety-critical actions: Rule-based override\n"
        "2. High DRL confidence: DRL override\n"
        "3. Low confidence: Rule-based default"
    )
    ax.text(6, 3.2, decision_text, ha='center', va='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add confidence measure
    confidence_text = "DRL Confidence = max(action probabilities)"
    ax.text(9.5, 3.5, confidence_text, ha='center', va='center', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add recovery cooldown note
    cooldown_text = "Recovery Cooldown\nPeriod: 20 steps"
    ax.text(3.5, 5, cooldown_text, ha='center', va='center', fontsize=9, color='gray',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Add hybrid agent label at the top
    plt.text(6, 8.5, "Hybrid Agent Architecture", fontsize=16, ha='center', va='bottom', weight='bold')
    
    # Configure the axes
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_architecture.png'))
    plt.close()
    print("  Hybrid architecture diagram saved to", os.path.join(OUTPUT_DIR, 'hybrid_architecture.png'))

def create_rule_based_flowchart():
    """
    Create Figure 8c: Rule-Based FDIR Logic Flowchart showing the decision tree used by the classical agent.
    
    # Paper reference: Section 3.2 "Classical FDIR Agent (RuleBasedFDIR)" - This flowchart
    # illustrates the deterministic nature of threshold-based fault detection and predefined
    # recovery actions in the classical approach.
    """
    print("Generating rule-based FDIR flowchart...")
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Define colors and shapes
    start_color = '#2ecc71'  # Green
    decision_color = '#3498db'  # Blue
    action_color = '#e74c3c'  # Red
    box_alpha = 0.7
    
    # Box properties
    box_width = 3.0
    box_height = 1.0
    
    # Define positions
    start_pos = (5, 11)
    
    # Start node
    start_box = Rectangle((start_pos[0]-box_width/2, start_pos[1]-box_height/2), 
                          box_width, box_height, facecolor=start_color, edgecolor='black', 
                          alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(start_box)
    ax.text(start_pos[0], start_pos[1], "Receive Observation", ha='center', va='center', fontweight='bold')
    
    # First decision - EPS voltage
    eps_pos = (5, 9)
    eps_box = Rectangle((eps_pos[0]-box_width/2, eps_pos[1]-box_height/2), 
                        box_width, box_height, facecolor=decision_color, edgecolor='black', 
                        alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(eps_box)
    ax.text(eps_pos[0], eps_pos[1], "EPS Voltage < Threshold?", ha='center', va='center')
    
    # Draw arrow from start to EPS check
    arrow = FancyArrowPatch(start_pos, eps_pos, arrowstyle='->', mutation_scale=15, 
                           color='black', linewidth=1.5, zorder=1)
    ax.add_patch(arrow)
    
    # EPS recovery action
    eps_action_pos = (8, 9)
    eps_action_box = Rectangle((eps_action_pos[0]-box_width/2, eps_action_pos[1]-box_height/2), 
                              box_width, box_height, facecolor=action_color, edgecolor='black', 
                              alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(eps_action_box)
    ax.text(eps_action_pos[0], eps_action_pos[1], "RecoverEPS", ha='center', va='center', fontweight='bold')
    
    # Draw arrow from EPS check to recovery
    eps_yes_arrow = FancyArrowPatch(eps_pos, eps_action_pos, arrowstyle='->', mutation_scale=15, 
                                   color='black', linewidth=1.5, zorder=1)
    ax.add_patch(eps_yes_arrow)
    ax.text((eps_pos[0]+eps_action_pos[0])/2, eps_pos[1]+0.3, "Yes", ha='center', va='center')
    
    # Second decision - ADCS error
    adcs_pos = (5, 7)
    adcs_box = Rectangle((adcs_pos[0]-box_width/2, adcs_pos[1]-box_height/2), 
                         box_width, box_height, facecolor=decision_color, edgecolor='black', 
                         alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(adcs_box)
    ax.text(adcs_pos[0], adcs_pos[1], "ADCS Error > Threshold?", ha='center', va='center')
    
    # Draw arrow from EPS to ADCS check
    eps_no_arrow = FancyArrowPatch(eps_pos, adcs_pos, arrowstyle='->', mutation_scale=15, 
                                  color='black', linewidth=1.5, zorder=1)
    ax.add_patch(eps_no_arrow)
    ax.text(eps_pos[0]-0.4, (eps_pos[1]+adcs_pos[1])/2, "No", ha='right', va='center')
    
    # ADCS recovery action
    adcs_action_pos = (8, 7)
    adcs_action_box = Rectangle((adcs_action_pos[0]-box_width/2, adcs_action_pos[1]-box_height/2), 
                               box_width, box_height, facecolor=action_color, edgecolor='black', 
                               alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(adcs_action_box)
    ax.text(adcs_action_pos[0], adcs_action_pos[1], "RecoverADCS", ha='center', va='center', fontweight='bold')
    
    # Draw arrow from ADCS check to recovery
    adcs_yes_arrow = FancyArrowPatch(adcs_pos, adcs_action_pos, arrowstyle='->', mutation_scale=15, 
                                    color='black', linewidth=1.5, zorder=1)
    ax.add_patch(adcs_yes_arrow)
    ax.text((adcs_pos[0]+adcs_action_pos[0])/2, adcs_pos[1]+0.3, "Yes", ha='center', va='center')
    
    # Third decision - TCS temperature
    tcs_pos = (5, 5)
    tcs_box = Rectangle((tcs_pos[0]-box_width/2, tcs_pos[1]-box_height/2), 
                        box_width, box_height, facecolor=decision_color, edgecolor='black', 
                        alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(tcs_box)
    ax.text(tcs_pos[0], tcs_pos[1], "TCS Temp > Threshold?", ha='center', va='center')
    
    # Draw arrow from ADCS to TCS check
    adcs_no_arrow = FancyArrowPatch(adcs_pos, tcs_pos, arrowstyle='->', mutation_scale=15, 
                                   color='black', linewidth=1.5, zorder=1)
    ax.add_patch(adcs_no_arrow)
    ax.text(adcs_pos[0]-0.4, (adcs_pos[1]+tcs_pos[1])/2, "No", ha='right', va='center')
    
    # TCS recovery action
    tcs_action_pos = (8, 5)
    tcs_action_box = Rectangle((tcs_action_pos[0]-box_width/2, tcs_action_pos[1]-box_height/2), 
                              box_width, box_height, facecolor=action_color, edgecolor='black', 
                              alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(tcs_action_box)
    ax.text(tcs_action_pos[0], tcs_action_pos[1], "RecoverTCS", ha='center', va='center', fontweight='bold')
    
    # Draw arrow from TCS check to recovery
    tcs_yes_arrow = FancyArrowPatch(tcs_pos, tcs_action_pos, arrowstyle='->', mutation_scale=15, 
                                   color='black', linewidth=1.5, zorder=1)
    ax.add_patch(tcs_yes_arrow)
    ax.text((tcs_pos[0]+tcs_action_pos[0])/2, tcs_pos[1]+0.3, "Yes", ha='center', va='center')
    
    # Default action (No-op)
    noop_pos = (5, 3)
    noop_box = Rectangle((noop_pos[0]-box_width/2, noop_pos[1]-box_height/2), 
                         box_width, box_height, facecolor=action_color, edgecolor='black', 
                         alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(noop_box)
    ax.text(noop_pos[0], noop_pos[1], "No-op", ha='center', va='center', fontweight='bold')
    
    # Draw arrow from TCS to No-op
    tcs_no_arrow = FancyArrowPatch(tcs_pos, noop_pos, arrowstyle='->', mutation_scale=15, 
                                  color='black', linewidth=1.5, zorder=1)
    ax.add_patch(tcs_no_arrow)
    ax.text(tcs_pos[0]-0.4, (tcs_pos[1]+noop_pos[1])/2, "No", ha='right', va='center')
    
    # Add title
    plt.text(5, 12.5, "Rule-Based FDIR Logic Flowchart", fontsize=16, ha='center', va='bottom', weight='bold')
    
    # Add "Return Action" box at the bottom
    return_pos = (5, 1)
    return_box = Rectangle((return_pos[0]-box_width/2, return_pos[1]-box_height/2), 
                           box_width, box_height, facecolor='#f39c12', edgecolor='black', 
                           alpha=box_alpha, linewidth=2, zorder=2)
    ax.add_patch(return_box)
    ax.text(return_pos[0], return_pos[1], "Return Selected Action", ha='center', va='center')
    
    # Connect all actions to return
    for action_pos in [eps_action_pos, adcs_action_pos, tcs_action_pos, noop_pos]:
        return_arrow = FancyArrowPatch(action_pos, (return_pos[0], action_pos[1]), arrowstyle='->', 
                                      linestyle=':', mutation_scale=15, color='black', linewidth=1, zorder=1)
        ax.add_patch(return_arrow)
        
        if action_pos != noop_pos:
            down_arrow = FancyArrowPatch((return_pos[0], action_pos[1]), return_pos, arrowstyle='->', 
                                        linestyle=':', mutation_scale=15, color='black', linewidth=1, zorder=1)
            ax.add_patch(down_arrow)
    
    # Add notes about static thresholds
    notes = "Static Thresholds:\n- EPS Voltage < 27.5V\n- ADCS Error > 0.15 rad\n- TCS Temp > 35°C"
    ax.text(2, 2, notes, ha='left', va='center', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Configure the axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rule_based_flowchart.png'))
    plt.close()
    print("  Rule-based flowchart saved to", os.path.join(OUTPUT_DIR, 'rule_based_flowchart.png'))

if __name__ == "__main__":
    print("Generating agent architecture diagrams...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    create_drl_architecture()
    create_hybrid_architecture()
    create_rule_based_flowchart()
    print("Architecture diagram generation complete!") 
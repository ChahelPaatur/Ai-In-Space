#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

# Output directory
OUTPUT_DIR = 'static/plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'paper'), exist_ok=True)

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

def create_sfri_components():
    """
    Generate Figure 11: SFRI Metric Components
    
    Illustrates how detection accuracy, recovery time, system stability impact, 
    and false positive penalties are combined into a single comprehensive metric.
    
    # Paper reference: Section 3.5 "Metrics Framework" - The SFRI weights and components
    # as described in the paper:
    # "SFRI = 35 × (DetectionRate) + 25 × (1 - MTTR/MaxSteps) + 10 × (StabilityScore) - 30 × (FalsePositiveRate)"
    """
    print("Generating SFRI components diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    detection_color = '#2ecc71'  # Green
    recovery_color = '#3498db'   # Blue
    stability_color = '#9b59b6'  # Purple
    falsepos_color = '#e74c3c'   # Red
    sfri_color = '#f39c12'       # Orange
    
    # Define box properties
    box_width = 2.5
    box_height = 1.5
    box_alpha = 0.8
    
    # Draw the component boxes
    # Detection Rate
    detection_pos = (3, 6)
    detection_box = Rectangle(
        (detection_pos[0] - box_width/2, detection_pos[1] - box_height/2),
        box_width, box_height, facecolor=detection_color, alpha=box_alpha,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(detection_box)
    ax.text(detection_pos[0], detection_pos[1], "Detection Rate\n(0-100%)", 
             ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    # Recovery Time
    recovery_pos = (7, 6)
    recovery_box = Rectangle(
        (recovery_pos[0] - box_width/2, recovery_pos[1] - box_height/2),
        box_width, box_height, facecolor=recovery_color, alpha=box_alpha,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(recovery_box)
    ax.text(recovery_pos[0], recovery_pos[1], "Recovery Time\n(MTTR)", 
             ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    # Stability Impact
    stability_pos = (3, 3)
    stability_box = Rectangle(
        (stability_pos[0] - box_width/2, stability_pos[1] - box_height/2),
        box_width, box_height, facecolor=stability_color, alpha=box_alpha,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(stability_box)
    ax.text(stability_pos[0], stability_pos[1], "System Stability\nScore", 
             ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    # False Positive Rate
    falsepos_pos = (7, 3)
    falsepos_box = Rectangle(
        (falsepos_pos[0] - box_width/2, falsepos_pos[1] - box_height/2),
        box_width, box_height, facecolor=falsepos_color, alpha=box_alpha,
        edgecolor='black', linewidth=1.5
    )
    ax.add_patch(falsepos_box)
    ax.text(falsepos_pos[0], falsepos_pos[1], "False Positive\nRate", 
             ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    # Final SFRI Metric
    sfri_pos = (5, 1)
    sfri_width = 4.0
    sfri_box = Rectangle(
        (sfri_pos[0] - sfri_width/2, sfri_pos[1] - box_height/2),
        sfri_width, box_height, facecolor=sfri_color, alpha=box_alpha,
        edgecolor='black', linewidth=2
    )
    ax.add_patch(sfri_box)
    ax.text(sfri_pos[0], sfri_pos[1], "SFRI Score\n(0-100 scale)", 
             ha='center', va='center', fontsize=14, weight='bold', color='white')
    
    # Draw arrows
    arrow_style = '-|>'
    mutation_scale = 20
    
    # Detection to SFRI
    detection_arrow = FancyArrowPatch(
        (detection_pos[0], detection_pos[1] - box_height/2 - 0.1),
        (sfri_pos[0] - sfri_width/4, sfri_pos[1] + box_height/2 + 0.1),
        connectionstyle="arc3,rad=0.1", 
        arrowstyle=arrow_style, color=detection_color, 
        linewidth=2, mutation_scale=mutation_scale
    )
    ax.add_patch(detection_arrow)
    
    # Recovery to SFRI
    recovery_arrow = FancyArrowPatch(
        (recovery_pos[0], recovery_pos[1] - box_height/2 - 0.1),
        (sfri_pos[0] + sfri_width/4, sfri_pos[1] + box_height/2 + 0.1),
        connectionstyle="arc3,rad=-0.1", 
        arrowstyle=arrow_style, color=recovery_color, 
        linewidth=2, mutation_scale=mutation_scale
    )
    ax.add_patch(recovery_arrow)
    
    # Stability to SFRI
    stability_arrow = FancyArrowPatch(
        (stability_pos[0], stability_pos[1] - box_height/2 - 0.1),
        (sfri_pos[0] - sfri_width/6, sfri_pos[1] + box_height/2 + 0.1),
        connectionstyle="arc3,rad=0.05", 
        arrowstyle=arrow_style, color=stability_color, 
        linewidth=2, mutation_scale=mutation_scale
    )
    ax.add_patch(stability_arrow)
    
    # False Positive to SFRI (negative impact)
    falsepos_arrow = FancyArrowPatch(
        (falsepos_pos[0], falsepos_pos[1] - box_height/2 - 0.1),
        (sfri_pos[0] + sfri_width/6, sfri_pos[1] + box_height/2 + 0.1),
        connectionstyle="arc3,rad=-0.05", 
        arrowstyle=arrow_style, color=falsepos_color, 
        linewidth=2, mutation_scale=mutation_scale
    )
    ax.add_patch(falsepos_arrow)
    
    # Add weight labels
    weight_fontsize = 12
    weight_bgcolor = 'white'
    
    # Detection weight
    ax.text(3.3, 4.7, "× 35%", fontsize=weight_fontsize, weight='bold', color=detection_color,
            bbox=dict(facecolor=weight_bgcolor, alpha=0.8, boxstyle='round,pad=0.3', edgecolor=detection_color))
    
    # Recovery weight
    ax.text(6.6, 4.7, "× 25%", fontsize=weight_fontsize, weight='bold', color=recovery_color,
            bbox=dict(facecolor=weight_bgcolor, alpha=0.8, boxstyle='round,pad=0.3', edgecolor=recovery_color))
    
    # Stability weight
    ax.text(3.3, 2.2, "× 10%", fontsize=weight_fontsize, weight='bold', color=stability_color,
            bbox=dict(facecolor=weight_bgcolor, alpha=0.8, boxstyle='round,pad=0.3', edgecolor=stability_color))
    
    # False positive weight (negative)
    ax.text(6.6, 2.2, "× -30%", fontsize=weight_fontsize, weight='bold', color=falsepos_color,
            bbox=dict(facecolor=weight_bgcolor, alpha=0.8, boxstyle='round,pad=0.3', edgecolor=falsepos_color))
    
    # Add formula at the top
    formula = "SFRI = 35 × (DetectionRate) + 25 × (1 - MTTR/MaxSteps) + 10 × (StabilityScore) - 30 × (FalsePositiveRate)"
    ax.text(5, 8, formula, fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Add component descriptions
    descrip_fontsize = 9
    ax.text(detection_pos[0], detection_pos[1] - box_height/2 - 0.4, 
            "Percentage of faults correctly\ndetected by the agent",
            ha='center', va='top', fontsize=descrip_fontsize,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(recovery_pos[0], recovery_pos[1] - box_height/2 - 0.4, 
            "Average time steps between fault\ninjection and successful recovery",
            ha='center', va='top', fontsize=descrip_fontsize,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(stability_pos[0], stability_pos[1] - box_height/2 - 0.4, 
            "Percentage of time system remains\nwithin nominal parameter ranges",
            ha='center', va='top', fontsize=descrip_fontsize,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    ax.text(falsepos_pos[0], falsepos_pos[1] - box_height/2 - 0.4, 
            "Ratio of false positive actions\nto total actions taken",
            ha='center', va='top', fontsize=descrip_fontsize,
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    
    # Add title
    ax.text(5, 9, "Stability Fault Recovery Index (SFRI) Components", 
            fontsize=16, ha='center', va='center', weight='bold')
    
    # Add explanation at the bottom
    explanation = (
        "The SFRI metric balances fault detection capability, recovery efficiency, overall system stability,\n"
        "and false positive penalties to provide a comprehensive evaluation of FDIR performance."
    )
    ax.text(5, 0, explanation, fontsize=10, ha='center', va='center', 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Configure the axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sfri_components.png'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'paper', 'figure11_sfri_components.png'))
    plt.close()
    
    print(f"  SFRI components diagram saved to {os.path.join(OUTPUT_DIR, 'paper', 'figure11_sfri_components.png')}")

if __name__ == "__main__":
    print("Generating SFRI Components Diagram (Figure 11)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'paper'), exist_ok=True)
    create_sfri_components()
    print("Figure 11 generation complete!") 
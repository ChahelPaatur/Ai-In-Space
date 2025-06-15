import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Create output directory
Path("static/plots/paper").mkdir(parents=True, exist_ok=True)

# Load p-value data
with open("P_value/sfri_pvalues.json", "r") as file:
    data = json.load(file)

# Extract data for plotting
comparisons = list(data.keys())
p_values = [-np.log10(data[comp]["p_value"]) for comp in comparisons]  # Use -log10 for better visualization
effect_sizes = [abs(data[comp]["effect_size"]) for comp in comparisons]
better_agents = [data[comp]["better_agent"] for comp in comparisons]

# Determine bar colors based on better agent
colors = []
for agent in better_agents:
    if agent == "classical":
        colors.append("blue")
    elif agent == "drl":
        colors.append("orange")
    elif agent == "hybrid":
        colors.append("green")

# Nicer labels for plotting
labels = [comp.replace("_", " vs ").title() for comp in comparisons]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot p-values (as -log10(p))
bars1 = ax1.bar(labels, p_values, color=colors)
ax1.set_title("Statistical Significance of SFRI Comparisons")
ax1.set_ylabel("-log10(p-value)")
ax1.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
ax1.axhline(y=-np.log10(0.01), color='darkred', linestyle='--', label='p=0.01')
ax1.legend()

# Annotate with actual p-values
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'p={data[comparisons[i]]["p_value"]:.2e}',
            ha='center', va='bottom', rotation=0, fontsize=9)

# Plot effect sizes
bars2 = ax2.bar(labels, effect_sizes, color=colors)
ax2.set_title("Effect Size of SFRI Comparisons")
ax2.set_ylabel("Cohen's d (absolute value)")
ax2.axhline(y=0.2, color='gray', linestyle='--', label='Small')
ax2.axhline(y=0.5, color='gray', linestyle='-', label='Medium')
ax2.axhline(y=0.8, color='black', linestyle='--', label='Large')
ax2.legend()

# Annotate with actual effect sizes and better agent
for i, bar in enumerate(bars2):
    height = bar.get_height()
    effect_size = data[comparisons[i]]["effect_size"]
    better = data[comparisons[i]]["better_agent"]
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'd={effect_size:.2f}\n{better}',
            ha='center', va='bottom', rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("static/plots/paper/figure_pvalues_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig("P_value/pvalues_visualization.png", dpi=300, bbox_inches='tight')
print("P-value visualization created and saved to static/plots/paper/figure_pvalues_comparison.png")

# Create a detailed tabular visualization
plt.figure(figsize=(10, 6))
plt.axis('off')

# Create table data
table_data = []
headers = ["Comparison", "P-Value", "Significant", "Effect Size", "Better Agent"]
for comp in comparisons:
    p_val = data[comp]["p_value"]
    sig = "Yes (p<0.001)" if data[comp]["significant_001"] else "Yes (p<0.05)" if data[comp]["significant_005"] else "No"
    effect = data[comp]["effect_size"]
    effect_mag = "Large" if abs(effect) >= 0.8 else "Medium" if abs(effect) >= 0.5 else "Small" if abs(effect) >= 0.2 else "Negligible"
    better = data[comp]["better_agent"]
    
    table_data.append([
        comp.replace("_", " vs ").title(),
        f"{p_val:.2e}",
        sig,
        f"{effect:.2f} ({effect_mag})",
        better.title()
    ])

# Plot table
table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Add title
plt.title("Statistical Analysis of SFRI Comparisons", pad=20, fontsize=14)

plt.tight_layout()
plt.savefig("static/plots/paper/figure_pvalues_table.png", dpi=300, bbox_inches='tight')
plt.savefig("P_value/pvalues_table.png", dpi=300, bbox_inches='tight')
print("P-value table visualization created and saved to static/plots/paper/figure_pvalues_table.png")

print("All p-value visualizations completed successfully.") 
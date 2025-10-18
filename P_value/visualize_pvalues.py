import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load p-value results
with open('P_value/sfri_pvalues.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
comparisons = ["classical_vs_drl", "classical_vs_hybrid", "drl_vs_hybrid"]
p_values = [data[comp]["p_value"] for comp in comparisons]
mean_diffs = [abs(data[comp]["mean_diff"]) for comp in comparisons]
t_statistics = [abs(data[comp]["t_statistic"]) for comp in comparisons]

# Create the plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: P-values with significance thresholds
ax1.bar(range(len(comparisons)), p_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='p = 0.05')
ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='p = 0.01') 
ax1.axhline(y=0.001, color='darkred', linestyle='--', alpha=0.7, label='p = 0.001')
ax1.set_yscale('log')
ax1.set_xlabel('Agent Comparison')
ax1.set_ylabel('P-value (log scale)')
ax1.set_title('Statistical Significance of SFRI Differences')
ax1.set_xticks(range(len(comparisons)))
ax1.set_xticklabels(['Classical vs DRL', 'Classical vs Hybrid', 'DRL vs Hybrid'], rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add significance annotations
for i, (comp, pval) in enumerate(zip(comparisons, p_values)):
    if pval < 0.001:
        ax1.annotate('***', xy=(i, pval), xytext=(i, pval*10), 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif pval < 0.01:
        ax1.annotate('**', xy=(i, pval), xytext=(i, pval*10), 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    elif pval < 0.05:
        ax1.annotate('*', xy=(i, pval), xytext=(i, pval*10), 
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

# Plot 2: Mean differences (effect sizes)
ax2.bar(range(len(comparisons)), mean_diffs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_xlabel('Agent Comparison')
ax2.set_ylabel('SFRI Score Difference (points)')
ax2.set_title('Magnitude of SFRI Score Differences')
ax2.set_xticks(range(len(comparisons)))
ax2.set_xticklabels(['Classical vs DRL', 'Classical vs Hybrid', 'DRL vs Hybrid'], rotation=45)
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for i, diff in enumerate(mean_diffs):
    ax2.text(i, diff + 0.5, f'{diff:.1f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('P_value/sfri_pvalue_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('static/plots/paper/figure12_pvalue_analysis.png', dpi=300, bbox_inches='tight')

print("P-value analysis plots saved to:")
print("  - P_value/sfri_pvalue_analysis.png")
print("  - static/plots/paper/figure12_pvalue_analysis.png")

# Create a summary table
print("\n=== STATISTICAL ANALYSIS SUMMARY ===")
print(f"{'Comparison':<20} {'P-value':<12} {'Significance':<15} {'Mean Diff':<10}")
print("-" * 65)
for comp in comparisons:
    p_val = data[comp]["p_value"]
    sig_level = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    mean_diff = abs(data[comp]["mean_diff"])
    comp_name = comp.replace("_", " ").title()
    print(f"{comp_name:<20} {p_val:<12.6f} {sig_level:<15} {mean_diff:<10.1f}")

plt.close() 
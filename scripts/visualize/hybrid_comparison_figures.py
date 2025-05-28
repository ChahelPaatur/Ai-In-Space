import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = "static/plots/hybrid_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set the style for paper-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
PAPER_STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6)
}
plt.rcParams.update(PAPER_STYLE)

# Results from the comparison
original_hybrid = {
    'mean_reward': -45.62,
    'false_positives': 857,
    'false_positive_rate': 0.96,
    'detection_rate': 70.0,
    'recovery_rate': 57.5,
    'mttd': 60.7,
    'mttr': 85.9,
    'sfri': 22.2
}

enhanced_hybrid = {
    'mean_reward': -38.61,
    'false_positives': 528,
    'false_positive_rate': 0.92,
    'detection_rate': 56.5,
    'recovery_rate': 37.0,
    'mttd': 88.1,
    'mttr': 127.2,
    'sfri': 18.3,
    'false_positives_prevented': 171
}

# Figure 1: Comparison of key metrics between original and enhanced hybrid agents
def create_key_metrics_comparison():
    # Metrics to compare
    metrics = ['False Positives', 'Detection Rate (%)', 'MTTD', 'SFRI']
    
    # Values for each metric
    original_values = [original_hybrid['false_positives'], 
                       original_hybrid['detection_rate'], 
                       original_hybrid['mttd'], 
                       original_hybrid['sfri']]
    
    enhanced_values = [enhanced_hybrid['false_positives'], 
                      enhanced_hybrid['detection_rate'], 
                      enhanced_hybrid['mttd'], 
                      enhanced_hybrid['sfri']]
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # For each metric, create a subplot
    for i, (metric, orig_val, enh_val) in enumerate(zip(metrics, original_values, enhanced_values)):
        ax = axs[i]
        
        # Create bars
        bars = ax.bar(['Original Hybrid', 'Enhanced Hybrid'], [orig_val, enh_val], 
                      color=['#4F81BD', '#C0504D'])
        
        # Add text above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02, 
                    f'{height:.1f}', ha='center', va='bottom')
        
        # Set title and labels
        ax.set_title(metric)
        ax.set_ylabel(metric)
        
        # Different y-axis for false positives (much larger values)
        if metric == 'False Positives':
            ax.set_ylim(0, max(orig_val, enh_val) * 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_key_metrics_comparison.png'), dpi=300)
    plt.close()

# Figure 2: Tradeoff visualization - Detection Rate vs False Positives
def create_tradeoff_visualization():
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot
    plt.scatter([original_hybrid['false_positives']], [original_hybrid['detection_rate']], 
                s=200, color='#4F81BD', label='Original Hybrid', marker='o')
    plt.scatter([enhanced_hybrid['false_positives']], [enhanced_hybrid['detection_rate']], 
                s=200, color='#C0504D', label='Enhanced Hybrid', marker='s')
    
    # Add labels to the points
    plt.annotate('Original Hybrid', 
                 xy=(original_hybrid['false_positives'], original_hybrid['detection_rate']),
                 xytext=(original_hybrid['false_positives']+20, original_hybrid['detection_rate']+2),
                 arrowprops=dict(arrowstyle='->'))
    
    plt.annotate('Enhanced Hybrid', 
                 xy=(enhanced_hybrid['false_positives'], enhanced_hybrid['detection_rate']),
                 xytext=(enhanced_hybrid['false_positives']-100, enhanced_hybrid['detection_rate']-5),
                 arrowprops=dict(arrowstyle='->'))
    
    # Draw an arrow showing the tradeoff direction
    plt.annotate('Tradeoff Direction', 
                 xy=((original_hybrid['false_positives']+enhanced_hybrid['false_positives'])/2,
                     (original_hybrid['detection_rate']+enhanced_hybrid['detection_rate'])/2),
                 xytext=((original_hybrid['false_positives']+enhanced_hybrid['false_positives'])/2+100,
                         (original_hybrid['detection_rate']+enhanced_hybrid['detection_rate'])/2-10),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # Add titles and labels
    plt.title('Detection Rate vs False Positives Tradeoff')
    plt.xlabel('False Positives')
    plt.ylabel('Detection Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits
    plt.xlim(min(enhanced_hybrid['false_positives']-50, 0), 
             original_hybrid['false_positives']+100)
    plt.ylim(enhanced_hybrid['detection_rate']-10, 
             original_hybrid['detection_rate']+10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_tradeoff_visualization.png'), dpi=300)
    plt.close()

# Figure 3: Bar chart of metrics with percent difference
def create_percent_difference_chart():
    # Metrics to compare
    metrics = ['False Positives', 'Detection Rate', 'MTTD', 'MTTR', 'SFRI']
    
    # Values for original hybrid
    original_values = [original_hybrid['false_positives'], 
                       original_hybrid['detection_rate'], 
                       original_hybrid['mttd'], 
                       original_hybrid['mttr'],
                       original_hybrid['sfri']]
    
    # Calculate percent differences
    percent_diffs = []
    for i, metric in enumerate(metrics):
        if metric == 'False Positives':
            # For false positives, reduction is positive
            pct = (original_hybrid['false_positives'] - enhanced_hybrid['false_positives']) / original_hybrid['false_positives'] * 100
        elif metric == 'Detection Rate':
            pct = (enhanced_hybrid['detection_rate'] - original_hybrid['detection_rate']) / original_hybrid['detection_rate'] * 100
        elif metric == 'MTTD':
            # For MTTD, increase is negative
            pct = (enhanced_hybrid['mttd'] - original_hybrid['mttd']) / original_hybrid['mttd'] * 100 * -1
        elif metric == 'MTTR':
            # For MTTR, increase is negative
            pct = (enhanced_hybrid['mttr'] - original_hybrid['mttr']) / original_hybrid['mttr'] * 100 * -1
        elif metric == 'SFRI':
            pct = (enhanced_hybrid['sfri'] - original_hybrid['sfri']) / original_hybrid['sfri'] * 100
        
        percent_diffs.append(pct)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create bars with colors based on positive/negative values
    colors = ['green' if x >= 0 else 'red' for x in percent_diffs]
    bars = plt.bar(metrics, percent_diffs, color=colors)
    
    # Add text above/below bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            va = 'bottom'
            offset = 1
        else:
            va = 'top'
            offset = -1
        
        plt.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{height:.1f}%', ha='center', va=va)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add titles and labels
    plt.title('Percent Change in Metrics: Enhanced vs Original Hybrid')
    plt.ylabel('Percent Change (%)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set y-axis limits
    max_abs = max(abs(min(percent_diffs)), abs(max(percent_diffs)))
    plt.ylim(-max_abs*1.1, max_abs*1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hybrid_percent_difference.png'), dpi=300)
    plt.close()

# Run all figure creation functions
if __name__ == "__main__":
    print("Creating Key Metrics Comparison figure...")
    create_key_metrics_comparison()
    
    print("Creating Tradeoff Visualization figure...")
    create_tradeoff_visualization()
    
    print("Creating Percent Difference Chart...")
    create_percent_difference_chart()
    
    print("All comparison figures created successfully!") 
import os
import glob
import json
from flask import Flask, render_template, url_for, jsonify, send_from_directory

app = Flask(__name__)

# Configuration
PLOTS_DIR = "plots"
SUMMARY_FILE = "results.json"
RESULTS_FILE = "results.json"
LOGS_DIR = "logs"
PLOT_DIR = "static/plots"

# Define numpy globally for get_summary_stats
try:
    import numpy as np
except ImportError:
    print("Warning: Numpy not found. Summary statistics calculation might fail.")
    np = None

# --- Explanations for Plots (Enhanced for Web with Structure) ---
EXPLANATIONS = {
    "reward_comparison": """
<h4 class='mb-3'>Analysis: Episode Rewards</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>This plot compares the overall success of each agent across multiple simulation runs (episodes). The 'reward' is a score calculated at each step, designed to encourage good performance (maintaining stability, recovering from faults) and penalize failures or inefficient operation. Higher total reward per episode indicates better overall performance.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
    <ul>
        <li>The <strong>DeepRL agent (orange)</strong> consistently achieves higher (less negative) total rewards compared to the <strong>Classical agent (blue)</strong>. The entire orange box, representing the middle 50% of DRL results, is above the classical box.</li>
        <li>The median reward (the line inside the box) for DeepRL is significantly better (closer to zero).</li>
        <li>Both agents show variability in rewards (height of the boxes), but the Classical agent has a wider spread and more extreme negative outliers (dots below the main whisker).</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>Negative rewards indicate that penalties (e.g., for active faults, instability) outweighed positive incentives in these specific, fault-heavy test episodes. However, the DRL agent was considerably more effective at mitigating these penalties, suggesting it learned a more robust strategy for handling the simulated faults and maintaining operational goals compared to the static rules of the classical agent.</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>This is strong evidence that the DeepRL approach, despite needing training, can lead to significantly improved operational resilience and performance in complex scenarios with potential faults, outperforming traditional rule-based systems in this simulated environment.</p>
</div>
""",
    "steps_comparison": """
<h4 class='mb-3'>Analysis: Episode Duration (Steps)</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>This plot compares how long each agent managed to keep the simulation running before it ended. An episode ends either by reaching the maximum allowed steps (200 in this case, called 'truncation') or by a critical failure ('termination'). Longer durations are generally better, indicating stability.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
     <ul>
        <li>Both agents consistently reach the maximum 200 steps in almost all displayed episodes (the boxes are tightly clustered at the top y-value of 200).</li>
        <li>There might be slight variations or outliers below 200, but the core performance shows both agents avoided catastrophic failures that would terminate the episode early in these specific runs.</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>In these particular test scenarios (5 episodes each), neither agent suffered a critical failure leading to early termination. Both operated until the simulation time limit. This suggests both were capable of basic survival, but it doesn't necessarily differentiate their finer control performance (which is better captured by the Rewards plot).</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>While demonstrating basic operational capability for both agents, this plot highlights the importance of looking beyond just survival time. The Rewards plot provides a more nuanced view of *how well* each agent performed *during* those 200 steps.</p>
</div>
""",
    "timeseries": """
<h4 class='mb-3'>Analysis: Telemetry Time Series (Example: Episode 0)</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>These plots provide a detailed, step-by-step look at how specific sensor readings or system states (like battery charge, attitude error, temperature) evolved during *one specific simulation run* (Episode 0). They overlay the behavior of the Classical and DeepRL agents under the *exact same* initial conditions and sequence of injected faults for that episode, allowing direct comparison.</p>
</div>
<div class='explanation-section'>
    <h5>Example Analysis (Temperature Plot)</h5>
    <ul>
        <li>Look for differences in the lines. Does one agent maintain a more stable value? Does one react more quickly or dramatically to changes?</li>
        <li>In the Temperature plot, notice how the DeepRL agent's temperature (orange line) sometimes shows sharp spikes and drops (e.g., around step 160), while the Classical agent's temperature (blue line) changes more smoothly.</li>
        <li>The smooth line for the Classical agent might indicate it's not actively managing temperature or its control rules are slow acting. The sharp changes for the DeepRL agent suggest it *is* taking actions (like turning a heater on or off, which the Action Distribution plot confirms it learned to do) in response to thermal conditions or other system states, even if those actions cause temporary fluctuations. This proactive or reactive behavior was learned, not pre-programmed.</li>
        <li>Similar analysis can be applied to State of Charge (SoC - battery management), Attitude Error (pointing accuracy), etc., revealing differences in resource management and control stability.</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>Time series plots are crucial for understanding the *how* and *why* behind the overall performance differences seen in the summary plots (Rewards, Steps). They reveal the moment-to-moment decision-making and dynamic responses of each agent to simulated events and faults.</p>
</div>
""",
    "action_distribution": """
<h4 class='mb-3'>Analysis: Action Choices</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>This histogram aggregates the choices made by each agent across all steps of all simulated episodes. It shows how frequently each possible control action (like doing nothing 'No-op', recovering a subsystem, turning heaters on/off) was selected.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
    <ul>
        <li><strong>No-op:</strong> Both agents spend a significant amount of time doing nothing, which is expected during nominal conditions. The Classical agent does this more often.</li>
        <li><strong>Recovery Actions:</strong> Both agents use `RecoverEPS`, `RecoverADCS`, `RecoverTCS`, but the counts differ. The DeepRL agent seems to use `RecoverTCS` slightly more often in this run.</li>
        <li><strong>DRL-Specific Actions:</strong> Critically, the DeepRL agent frequently uses actions like `HeaterON`, `ResetGyroBias`, and `EnterNominal`, which the Classical agent *never* used in these 5 episodes (indicated by the missing blue bars).</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>The Classical agent relies heavily on its pre-programmed recovery rules and defaults to No-op otherwise. The DeepRL agent learned a more diverse strategy. It learned to use heater controls, potentially for proactive thermal management, and to reset the gyro bias, possibly finding it beneficial for attitude control even when not strictly required by a simple rule. It also learned to actively enter the 'Nominal' state.</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>This plot visually demonstrates the core difference between the approaches. The Classical agent is limited by its explicit rules, while the DeepRL agent can discover and utilize a wider range of actions if its training experience indicated they lead to better long-term rewards (improved stability, faster recovery, better resource management).</p>
</div>
""",
    "learning_curve": """
<h4 class='mb-3'>Analysis: DRL Agent Learning Curve</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>This plot shows the DeepRL agent's learning progress during the training phase (`train_drl.py`). The X-axis represents the total simulation steps experienced by the agent, and the Y-axis shows the reward obtained in each episode, smoothed using a rolling average (typically over 100 episodes) to reduce noise and reveal the learning trend.</p>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>An upward trend indicates the agent is successfully learning to achieve higher rewards (fewer penalties). A flat curve might suggest the agent has converged (reached its peak performance for the current settings) or is stuck. A noisy curve with no clear trend might indicate instability or insufficient training time. Ideally, we want to see the reward increase and stabilize at a high value.</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>The learning curve is crucial for diagnosing the training process. It helps determine if more training time is needed, if hyperparameters require tuning, or if there are issues with the reward signal or environment complexity. It provides confidence (or lack thereof) in the agent's learned policy.</p>
</div>
"""
}
# --- End Explanations ---

def get_plot_files(directory):
    """Finds all .html plot files in the specified directory."""
    plot_files = {}
    search_path = os.path.join(directory, "*.html")
    # Use relative path for linking if serving directly
    # If using send_from_directory, keep as is.
    files = glob.glob(search_path)
    # Sort files for consistent order
    files.sort()
    for f in files:
        basename = os.path.basename(f)
        name = basename.replace('.html', '').replace('_', ' ').replace('Ep0', 'Ep 0').title()
        # Create a URL for the plot using Flask's url_for (requires a route)
        # Alternatively, create a relative link path if plots/ is served statically
        plot_files[name] = {'filename': basename, 'url': url_for('serve_plot', filename=basename)}
    return plot_files

def get_summary_stats(filename):
    """Loads summary statistics from the results JSON file."""
    stats = {'classical': None, 'drl': None}
    if not np: # Check if numpy import failed
        print("Cannot calculate summary stats because Numpy is not available.")
        return stats
        
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            for agent in ['classical', 'drl']:
                if agent in results and 'rewards' in results[agent] and 'steps' in results[agent]:
                    rewards = results[agent]['rewards']
                    steps = results[agent]['steps']
                    if rewards and steps: # Ensure lists are not empty
                         stats[agent] = {
                             'avg_reward': f"{np.mean(rewards):.2f} +/- {np.std(rewards):.2f}",
                             'avg_steps': f"{np.mean(steps):.2f} +/- {np.std(steps):.2f}",
                             'num_episodes': len(rewards)
                         }
        except Exception as e:
            print(f"Error loading or processing summary file {filename}: {e}")
    return stats

@app.route('/')
def index():
    """Main route to display the project overview and links to plots."""
    plot_files = get_plot_files(PLOTS_DIR)
    summary_stats = get_summary_stats(SUMMARY_FILE) 
    
    project_title = "Effectiveness of AI vs. Classical FDIR in Space Systems"
    project_abstract = ("Investigating the efficacy of Deep Reinforcement Learning (DRL) "
                        "for autonomous Fault Detection, Identification, and Recovery (FDIR) "
                        "in simulated spacecraft compared to classical rule-based methods.")

    return render_template('index.html', 
                           title=project_title,
                           abstract=project_abstract,
                           plot_files=plot_files,
                           summary_stats=summary_stats)

@app.route('/results')
def show_results():
    """Serves the visualization results page."""
    plot_files = {}
    plot_info = {}
    
    # General plots
    reward_plot = "episode_rewards_comparison_log.png"
    steps_plot = "episode_steps_comparison_log.png"
    action_plot = "action_distribution.png"
    
    if os.path.exists(os.path.join(PLOT_DIR, reward_plot)):
        plot_files['reward_comparison'] = reward_plot
        plot_info['reward_comparison'] = EXPLANATIONS['reward_comparison']
        
    if os.path.exists(os.path.join(PLOT_DIR, steps_plot)):
        plot_files['steps_comparison'] = steps_plot
        plot_info['steps_comparison'] = EXPLANATIONS['steps_comparison']
        
    if os.path.exists(os.path.join(PLOT_DIR, action_plot)):
        plot_files['action_distribution'] = action_plot
        plot_info['action_distribution'] = EXPLANATIONS['action_distribution']

    # Find time series plots (e.g., for episode 0)
    # Example: Find ep0_SoC_timeseries.png, ep0_AttErr_deg_timeseries.png, etc.
    # For simplicity, we'll just find *any* time series plots in the dir
    # A more robust approach would parse visualize_results output or logs
    timeseries_plots = glob.glob(os.path.join(PLOT_DIR, "ep*_timeseries.png"))
    plot_files['timeseries'] = [os.path.basename(p) for p in timeseries_plots]
    # Use the general explanation for all timeseries plots found
    if timeseries_plots:
         plot_info['timeseries'] = EXPLANATIONS['timeseries']

    # Check if any plots were found
    plots_exist = any(plot_files.values())

    # Add learning curve plot if it exists
    learning_curve_plot = "learning_curve.png"
    learning_curve_path = os.path.join(PLOT_DIR, learning_curve_plot)
    if os.path.exists(learning_curve_path):
        plot_files['learning_curve'] = learning_curve_plot
        plot_info['learning_curve'] = EXPLANATIONS['learning_curve']
        # Add flag to template context if learning curve exists
        # plots_exist = True # Ensure page shows something if only LC exists
    else:
        plot_files['learning_curve'] = None # Indicate it's missing

    return render_template(
        'results.html',
        title='Simulation Results',
        plots=plot_files,
        explanations=plot_info,
        plots_exist=plots_exist
    )

# Route to serve plot files from the 'plots' directory
@app.route('/plots/<path:filename>') # Use path converter for flexibility
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename, as_attachment=False)

# Example API endpoint (optional)
@app.route('/api/summary')
def api_summary():
    """Returns the summary results from results.json."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": f"Failed to load results: {str(e)}"}), 500
    else:
        return jsonify({"error": "Results file not found."}), 404

if __name__ == '__main__':
    # Use host='0.0.0.0' to make accessible on local network
    # Use debug=True for development (auto-reloads, provides debugger)
    app.run(debug=True, host='0.0.0.0', port=5001) 
import os
import glob
import json
from flask import Flask, render_template, url_for, jsonify, send_from_directory

app = Flask(__name__)

# --- Configuration ---
LOGS_DIR = "logs" # Directory containing raw simulation logs
PLOT_DIR = "static/plots" # Directory where generated plots are saved (for web serving)
SUMMARY_FILE = "results/comparison_summary.json" # Summary statistics file from run_comparison.py

# Dynamically import numpy for summary stats if available
try:
    import numpy as np
except ImportError:
    print("Warning: Numpy not found. Summary statistics calculation will fail.")
    np = None

# --- Explanations for Plots (HTML content for the results page) ---
# These multi-line strings contain HTML snippets providing context and interpretation
# for each plot type displayed on the /results page.
EXPLANATIONS = {
    "reward_comparison": """
<h4 class='mb-3'>Analysis: Episode Rewards</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>Compares total reward accumulated per episode. Higher rewards indicate better performance in achieving objectives and mitigating penalties.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
    <ul>
        <li>Compare median lines, box positions (interquartile range), and whisker lengths/outliers.</li>
        <li>Note which agent generally achieves higher (less negative) rewards.</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>Discuss why one agent might score better based on its strategy (e.g., effective fault mitigation vs. simple rules). Negative rewards indicate penalties dominated.</p>
</div>
""", # Simplified reward explanation
    "steps_comparison": """
<h4 class='mb-3'>Analysis: Episode Duration (Steps)</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>Compares episode lengths. Reaching the max steps (e.g., 200) indicates survival; shorter episodes imply early termination due to critical failure.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
     <ul>
        <li>Observe if agents consistently reach the maximum steps or if there are early terminations.</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>Consistent max steps suggests basic survival capability for both. Differences in reward are needed to assess performance *during* survival.</p>
</div>
""", # Simplified steps explanation
    "timeseries": """
<h4 class='mb-3'>Analysis: Telemetry Time Series (Example: Episode 0)</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>Shows step-by-step evolution of specific telemetry values (e.g., SoC, Temp, Attitude Error) for a single episode, comparing both agents under identical conditions.</p>
</div>
<div class='explanation-section'>
    <h5>Example Analysis (Temperature Plot)</h5>
    <ul>
        <li>Compare line stability, reaction speed, and magnitude of changes.</li>
        <li>Sharp changes might indicate active control (e.g., DRL using heaters), while smooth lines might indicate passive behavior or slower rules (Classical).</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>Reveals the dynamic behavior and moment-to-moment decision-making underlying the aggregate reward/step results.</p>
</div>
""", # Simplified timeseries explanation
    "action_distribution": """
<h4 class='mb-3'>Analysis: Action Choices</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>Histogram showing the frequency of each discrete action selected by agents across all evaluation steps.</p>
</div>
<div class='explanation-section'>
    <h5>Observations</h5>
    <ul>
        <li>Compare frequencies for common actions (No-op, basic recoveries).</li>
        <li>Identify actions used exclusively by one agent (e.g., DRL using HeaterON, ResetGyroBias).</li>
    </ul>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>Highlights differences in control strategies: fixed rules vs. learned, potentially more diverse behaviors.</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>Visually demonstrates the behavioral differences learned/programmed into each agent.</p>
</div>
""", # Simplified action distribution explanation
    "learning_curve": """
<h4 class='mb-3'>Analysis: DRL Agent Learning Curve</h4>
<div class='explanation-section'>
    <h5>Context</h5>
    <p>Shows DRL training progress: rolling average episode reward vs. total simulation steps experienced.</p>
</div>
<div class='explanation-section'>
    <h5>Interpretation</h5>
    <p>Upward trend indicates learning. Flat curve suggests convergence or stagnation. Noise indicates variability. Assess if training was sufficient.</p>
</div>
<div class='explanation-section'>
    <h5>Significance</h5>
    <p>Diagnoses the training process: confirms learning occurred, indicates if more training/tuning is needed.</p>
</div>
""" # Simplified learning curve explanation
}
# --- End Explanations ---

def get_summary_stats(filename):
    """Loads summary statistics (avg reward, steps) from the comparison JSON file."""
    stats = {'classical': None, 'drl': None}
    if not np: # Check numpy availability
        print("Cannot calculate summary stats: Numpy not available.")
        return stats

    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                results = json.load(f)
            # Calculate mean/stddev if data exists for both agents
            for agent in ['classical', 'drl']:
                if agent in results and results[agent].get('rewards') and results[agent].get('steps'):
                    rewards = results[agent]['rewards']
                    steps = results[agent]['steps']
                    stats[agent] = {
                        'avg_reward': f"{np.mean(rewards):.2f} ± {np.std(rewards):.2f}",
                        'avg_steps': f"{np.mean(steps):.2f} ± {np.std(steps):.2f}",
                        'num_episodes': len(rewards)
                    }
        except Exception as e:
            print(f"Error loading or processing summary file {filename}: {e}")
    else:
        print(f"Warning: Summary file not found at {filename}")
    return stats

@app.route('/')
def index():
    """Serves the main index page with project overview and links."""
    # Note: get_plot_files function was removed as it's not used by index/results
    # The results route now directly checks for specific plot files.
    summary_stats = get_summary_stats(SUMMARY_FILE)

    project_title = "AI vs. Classical FDIR: A Comparative Simulation Study"
    project_abstract = ("Investigating the efficacy of Deep Reinforcement Learning (DRL) "
                        "for autonomous Fault Detection, Identification, and Recovery (FDIR) "
                        "in simulated spacecraft compared to classical rule-based methods.")

    # Pass data to the Jinja2 template
    return render_template('index.html',
                           title=project_title,
                           abstract=project_abstract,
                           summary_stats=summary_stats)

@app.route('/results')
def show_results():
    """Serves the visualization results page, finding and displaying plots."""
    plot_files = {} # Stores filenames of existing plots
    plot_info = {} # Stores HTML explanations for existing plots
    plots_exist = False # Flag to indicate if any plots were found

    # Check for specific primary plot files
    primary_plots = {
        "reward_comparison": "episode_rewards_comparison_log.png",
        "steps_comparison": "episode_steps_comparison_log.png", # Keeping steps plot logic
        "action_distribution": "action_distribution.png",
        "learning_curve": "learning_curve.png",
    }

    for key, filename in primary_plots.items():
        filepath = os.path.join(PLOT_DIR, filename)
        if os.path.exists(filepath):
            plot_files[key] = filename
            plot_info[key] = EXPLANATIONS.get(key, f"<p>Explanation needed for {key}.</p>") # Use explanation if available
            plots_exist = True
        # Explicitly add missing plots with None value if needed by template logic
        # else:
        #     plot_files[key] = None

    # Find all time series plots (e.g., ep0_SoC_timeseries.png)
    timeseries_basenames = []
    try:
        # Use glob to find files matching the pattern
        timeseries_pattern = os.path.join(PLOT_DIR, "ep*_timeseries.png")
        timeseries_paths = glob.glob(timeseries_pattern)
        timeseries_basenames = [os.path.basename(p) for p in timeseries_paths]
        # Sort for consistent ordering
        timeseries_basenames.sort()
    except Exception as e:
         print(f"Error finding timeseries plots: {e}")

    if timeseries_basenames:
        plot_files['timeseries'] = timeseries_basenames
        plot_info['timeseries'] = EXPLANATIONS.get('timeseries', "<p>Explanation needed for timeseries.</p>")
        plots_exist = True

    # Pass found plots and explanations to the template
    return render_template(
        'results.html',
        title='Simulation Results Comparison',
        plots=plot_files,
        explanations=plot_info,
        plots_exist=plots_exist
    )


# This route is unnecessary if static files are configured correctly
# @app.route('/plots/<path:filename>')
# def serve_plot(filename):
#    # Flask can serve static files automatically if the `static_folder` is set
#    # or configured through web server (like Nginx/Apache)
#    # return send_from_directory(PLOT_DIR, filename)
#    pass # Remove or comment out if using standard static file serving

# Example API endpoint (optional, can be removed if not used)
# @app.route('/api/summary')
# def api_summary():
#    summary = get_summary_stats(SUMMARY_FILE)
#    return jsonify(summary)

if __name__ == '__main__':
    # Ensure the plot directory exists for static serving
    if not os.path.exists(PLOT_DIR):
        print(f"Warning: Plot directory '{PLOT_DIR}' not found. Creating it.")
        os.makedirs(PLOT_DIR, exist_ok=True) # Create if doesn't exist

    # Run the Flask development server
    # Debug=True enables auto-reloading and provides detailed error pages
    # Set host='0.0.0.0' to make it accessible on the network
    app.run(debug=True, host='0.0.0.0', port=5001) 
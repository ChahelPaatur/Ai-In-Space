import json
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

RESULTS_FILE = "results.json"
LOGS_DIR = "logs"
OUTPUT_DIR = "static/plots"

# --- Explanations for Plots ---
EXPLANATIONS = {
    "reward_comparison": """
### Comparison of Episode Rewards (Box Plot)
*   **What it shows:** Compares the distribution of the *total reward accumulated* in each full episode for the Classical agent vs. the DRL agent.
*   **Analysis:** Look at the median (line inside the box), the spread (box height/IQR), and outliers (dots). This helps compare typical performance, consistency, and best/worst-case scenarios between the agents.
""",
    "steps_comparison": """
### Comparison of Episode Lengths (Box Plot)
*   **What it shows:** Compares the distribution of the *total number of steps* (duration) each agent lasted per episode.
*   **Analysis:** Indicates agent longevity or survival time. Higher values are generally better, unless hitting a fixed maximum step limit. Compare median duration and variability.
""",
    "timeseries": """
### Telemetry Time Series (Line Plot - Episode Specific)
*   **What it shows:** Plots key satellite telemetry (like SoC, Attitude Error, Voltage) over time for a *single, specific episode*, overlaying the Classical and DRL agent trajectories for direct comparison under the same conditions.
*   **Analysis:** Shows how each agent manages resources (e.g., battery SoC), maintains control (e.g., attitude error), and responds to events within that specific episode instance.
""",
    "action_distribution": """
### Action Distribution (Histogram)
*   **What it shows:** Aggregates data *across all episodes* to show how frequently each discrete action was chosen by the Classical vs. the DRL agent.
*   **Analysis:** Reveals the overall control *strategy* or *policy*. Compare action preferences, frequency of recovery actions, or actions uniquely used by one agent.
"""
}
# --- End Explanations ---

def load_or_mock_results(filename=RESULTS_FILE):
    """Loads results from json file, or creates mock data if file doesn't exist."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                print(f"Loading results from {filename}")
                results = json.load(f)
                # Basic validation
                if 'classical' not in results or 'drl' not in results:
                    print("Warning: Results file missing classical or drl keys. Using mock data.")
                    return create_mock_results()
                return results
        except Exception as e:
            print(f"Error loading {filename}: {e}. Using mock data.")
            return create_mock_results()
    else:
        print(f"Results file {filename} not found. Creating mock data.")
        return create_mock_results()

def create_mock_results(num_episodes=20):
    """Creates mock results for demonstration purposes."""
    mock_data = {
        'classical': {
            'rewards': (np.random.randn(num_episodes) * 20 - 50).tolist(), # Lower avg reward
            'steps': (np.random.randint(150, 201, size=num_episodes)).tolist()
        },
        'drl': {
            'rewards': (np.random.randn(num_episodes) * 15 - 10).tolist(), # Higher avg reward
            'steps': (np.random.randint(180, 201, size=num_episodes)).tolist() # Maybe slightly longer avg steps?
        }
    }
    # Ensure DRL rewards are generally better than classical in mock data
    drl_avg = np.mean(mock_data['drl']['rewards'])
    classical_avg = np.mean(mock_data['classical']['rewards'])
    if drl_avg <= classical_avg:
        mock_data['drl']['rewards'] = (np.array(mock_data['drl']['rewards']) + (classical_avg - drl_avg) + 20).tolist()

    return mock_data

def load_detailed_logs(log_dir=LOGS_DIR):
    """Loads all episode logs from the specified directory."""
    all_logs = {'classical': [], 'drl': []}
    classical_files = glob.glob(os.path.join(log_dir, "classical_episode_*.json"))
    drl_files = glob.glob(os.path.join(log_dir, "drl_episode_*.json"))

    print(f"Found {len(classical_files)} classical logs and {len(drl_files)} DRL logs in '{log_dir}'.")

    # Map internal keys to display labels
    agent_label_map = {'classical': 'Classical', 'drl': 'DeepRL'}

    for agent_type, files in [('classical', classical_files), ('drl', drl_files)]:
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    episode_data = json.load(f)
                    # Add agent type and episode number for easier processing
                    try:
                         episode_num = int(os.path.basename(filepath).split('_')[-1].split('.')[0])
                    except ValueError:
                         episode_num = -1 # Assign default if parsing fails
                         
                    for step_data in episode_data:
                        step_data['Agent'] = agent_label_map.get(agent_type, agent_type.capitalize())
                        step_data['Episode'] = episode_num
                    all_logs[agent_type].extend(episode_data)
            except Exception as e:
                print(f"Error loading or processing log file {filepath}: {e}")
                
    if not all_logs['classical'] and not all_logs['drl']:
        print(f"No valid detailed logs found in {log_dir}. Cannot generate detailed plots.")
        return None
        
    # Convert combined list of step dictionaries into a DataFrame
    all_steps_list = []
    for agent_type, logs in all_logs.items():
        label = agent_label_map.get(agent_type, agent_type.capitalize())
        for step_data in logs:
            step_data['Agent'] = label # Assign the desired label
        all_steps_list.extend(logs)
        
    if not all_steps_list:
        return None

    df = pd.DataFrame(all_steps_list)
    # Ensure correct data types if loaded from JSON
    df['step'] = pd.to_numeric(df['step'])
    df['reward'] = pd.to_numeric(df['reward'])
    df['action'] = pd.to_numeric(df['action'])
    df['Episode'] = pd.to_numeric(df['Episode'])
    return df

def plot_summary_from_df(df, output_dir=OUTPUT_DIR, report_file=None):
    """Generates summary box plots from the combined step DataFrame and writes explanations to report."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot summary.")
        return
        
    # Calculate per-episode rewards and steps from the detailed logs
    episode_summary = df.groupby(['Agent', 'Episode']).agg(
        Reward=('reward', 'sum'),
        Steps=('step', 'max') # Max step number in the episode
    ).reset_index()

    if episode_summary.empty:
        print("Could not generate episode summary from logs.")
        return
        
    # Box Plot of Episode Rewards
    fig_rewards = px.box(episode_summary, x='Agent', y='Reward',
                         title="Comparison of Episode Rewards (from Logs)",
                         labels={'Reward': 'Total Reward per Episode'}, points="all", color='Agent',
                         hover_data=['Episode'])
    fig_rewards.update_layout(showlegend=False)
    reward_plot_file = os.path.join(output_dir, "episode_rewards_comparison_log.png")
    try:
        fig_rewards.write_image(reward_plot_file, scale=2)
        print(f"Saved reward comparison plot to {reward_plot_file}")
        if report_file:
            report_file.write(EXPLANATIONS["reward_comparison"])
            # Use relative path for Markdown link within the same output dir
            relative_reward_path = os.path.basename(reward_plot_file)
            report_file.write(f"![Reward Comparison Plot]({relative_reward_path})\n\n")
    except Exception as e:
         print(f"Error saving image {reward_plot_file}. Ensure kaleido is installed. Error: {e}")

    # Box Plot of Episode Steps
    fig_steps = px.box(episode_summary, x='Agent', y='Steps',
                       title="Comparison of Episode Lengths (from Logs)",
                       labels={'Steps': 'Steps per Episode'}, points="all", color='Agent',
                       hover_data=['Episode'])
    fig_steps.update_layout(showlegend=False)
    steps_plot_file = os.path.join(output_dir, "episode_steps_comparison_log.png")
    try:
        fig_steps.write_image(steps_plot_file, scale=2)
        print(f"Saved steps comparison plot to {steps_plot_file}")
        if report_file:
            report_file.write(EXPLANATIONS["steps_comparison"])
            relative_steps_path = os.path.basename(steps_plot_file)
            report_file.write(f"![Steps Comparison Plot]({relative_steps_path})\n\n")
    except Exception as e:
         print(f"Error saving image {steps_plot_file}. Ensure kaleido is installed. Error: {e}")

def plot_telemetry_timeseries(df, episode_to_plot=0, output_dir=OUTPUT_DIR, report_file=None):
    """Plots key telemetry over time, comparing agents, and writes explanation to report."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot time series.")
        return

    # Filter data for the chosen episode
    episode_df = df[df['Episode'] == episode_to_plot].copy()
    if episode_df.empty:
        print(f"No data found for episode {episode_to_plot}.")
        # Try plotting episode 0 if the requested one doesn't exist
        available_episodes = df['Episode'].unique()
        if episode_to_plot != 0 and 0 in available_episodes:
             print("Attempting to plot episode 0 instead.")
             plot_telemetry_timeseries(df, 0, output_dir)
        elif available_episodes.size > 0:
            first_episode = available_episodes[0]
            print(f"Attempting to plot first available episode ({first_episode}) instead.")
            plot_telemetry_timeseries(df, first_episode, output_dir)
        return

    print(f"Generating time series plots for Episode {episode_to_plot}...")

    # Expand the 'observation' list into separate columns
    # Indices map (based on SpacecraftEnv definition):
    # 0: SoC, 1: P_pot, 2: V_bus, 3: ChargeRate
    # 4-7: Quaternions, 8-10: AngVel, 11: AttErr
    # 12: TempA, 13: TempB, 14: HeaterStatus
    try:
        # Check if observation column actually contains lists
        if not episode_df['observation'].apply(isinstance, args=(list,)).all():
             raise ValueError("'observation' column does not contain lists.")
             
        obs_df = pd.DataFrame(episode_df['observation'].tolist(), 
                              columns=['SoC', 'P_pot', 'V_bus', 'ChargeRate', 
                                     'q1','q2','q3','q0', 'wx','wy','wz', 'AttErr',
                                     'TempA', 'TempB', 'HeaterStatus'], 
                              index=episode_df.index)
        # Convert Attitude Error to degrees for easier interpretation
        obs_df['AttErr_deg'] = np.degrees(obs_df['AttErr'])
        episode_df = pd.concat([episode_df.drop('observation', axis=1), obs_df], axis=1)
    except Exception as e:
        print(f"Error expanding observation column: {e}. Check log data structure.")
        # print("Sample observation data:", episode_df['observation'].head()) # Debug print
        return

    # Plot selected telemetry
    telemetry_to_plot = {
        'SoC': 'EPS State of Charge (%)',
        'V_bus': 'EPS Bus Voltage (V)',
        'ChargeRate': 'EPS Charge/Discharge Rate (W)',
        'AttErr_deg': 'ADCS Attitude Error (deg)',
        'TempA': 'TCS Temperature A (C)',
    }

    for col, title_suffix in telemetry_to_plot.items():
        if col in episode_df.columns:
            fig = px.line(episode_df, x='step', y=col, color='Agent',
                          title=f"Episode {episode_to_plot}: {title_suffix}",
                          labels={'step': 'Simulation Step', col: title_suffix.split(' (')[0]})
            # Add markers for fault events? (Requires parsing fault info)
            fig.update_layout(legend_title_text='Agent')
            plot_file = os.path.join(output_dir, f"ep{episode_to_plot}_{col}_timeseries.png")
            try:
                 fig.write_image(plot_file, scale=2)
                 print(f"Saved plot to {plot_file}")
                 if report_file:
                     # Write the specific plot image link
                     relative_plot_path = os.path.basename(plot_file)
                     plot_title = f"Episode {episode_to_plot}: {title_suffix}"
                     report_file.write(f"#### {plot_title}\n")
                     report_file.write(f"![{plot_title}]({relative_plot_path})\n\n")
            except Exception as e:
                 print(f"Error saving image {plot_file}. Ensure kaleido is installed. Error: {e}")
        else:
             print(f"Warning: Column '{col}' not found in log data.")
             
def plot_action_distribution(df, output_dir=OUTPUT_DIR, report_file=None):
    """Plots the distribution of actions taken by each agent."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot actions.")
        return
        
    # Action mapping (from SpacecraftEnv)
    action_labels = {
        0: 'No-op', 1: 'RecoverEPS', 2: 'RecoverADCS', 3: 'RecoverTCS', 
        4: 'HeaterON', 5: 'HeaterOFF', 6: 'ResetGyroBias', 
        7: 'EnterSafe', 8: 'EnterNominal'
    }
    # Map action numbers to labels for plotting
    df['ActionLabel'] = df['action'].map(action_labels).fillna('Unknown')
    
    fig = px.histogram(df, x='ActionLabel', color='Agent', barmode='group',
                       title="Distribution of Actions Taken (All Episodes)",
                       labels={'ActionLabel': 'Action Taken'})
    fig.update_layout(xaxis={'categoryorder':'array', 'categoryarray':[action_labels[i] for i in sorted(action_labels)]},
                      yaxis_title="Count")
    action_plot_file = os.path.join(output_dir, "action_distribution.png")
    try:
        fig.write_image(action_plot_file, scale=2)
        print(f"Saved action distribution plot to {action_plot_file}")
        if report_file:
            report_file.write(EXPLANATIONS["action_distribution"])
            relative_action_path = os.path.basename(action_plot_file)
            report_file.write(f"![Action Distribution Plot]({relative_action_path})\n\n")
    except Exception as e:
         print(f"Error saving image {action_plot_file}. Ensure kaleido is installed. Error: {e}")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_filename = os.path.join(OUTPUT_DIR, "plots_report.md")
    report_f = None # Initialize file handle

    try:
        report_f = open(report_filename, 'w')
        report_f.write("# Simulation Results Visualization Report\n\n")
        print(f"Generating plots and report in '{OUTPUT_DIR}' directory...")
        print(f"Report will be saved to: {report_filename}") # Added print statement

        # Load detailed logs into a single DataFrame
        detailed_df = load_detailed_logs()

        if detailed_df is not None:
            # Generate summary plots from the detailed data
            # Pass the file handle to the plotting functions
            plot_summary_from_df(detailed_df, output_dir=OUTPUT_DIR, report_file=report_f)

            # Generate time series plots for a specific episode
            available_episodes = sorted(detailed_df['Episode'].unique())
            valid_episodes = [ep for ep in available_episodes if ep >= 0]
            episode_to_show = valid_episodes[0] if valid_episodes else -1

            if episode_to_show != -1:
                plot_telemetry_timeseries(detailed_df, episode_to_show, output_dir=OUTPUT_DIR, report_file=report_f)
            else:
                print("No valid episodes found in logs to plot time series.")
                if report_f:
                    report_f.write("_No valid episodes found in logs to plot time series._\n\n")

            # Generate action distribution plot
            plot_action_distribution(detailed_df, output_dir=OUTPUT_DIR, report_file=report_f)

            report_f.write("\n---\n*Report generated automatically.*")
            print(f"\nReport successfully generated: {report_filename}")

        else:
             print("Could not load detailed logs. Visualization skipped.")
             if report_f:
                 report_f.write("_Could not load detailed logs. Visualization skipped._\n")

    except IOError as e:
        print(f"Error writing report file {report_filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during visualization: {e}")
    finally:
        if report_f:
            report_f.close() # Ensure file is closed even if errors occur


    # Optionally, load and plot summary from results.json as before (for comparison)
    # print("\nPlotting summary from results.json (if available)...")
    # from run_comparison import load_or_mock_results # Avoid circular import if run separately
    # summary_results = load_or_mock_results(RESULTS_FILE)
    # plot_results(summary_results) # Original plotting function if kept separate 
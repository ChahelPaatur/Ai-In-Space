import json
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- Configuration ---
RESULTS_FILE = "results/comparison_summary.json" # Input: Summary stats from run_comparison.py
LOGS_DIR = "logs"             # Input: Directory containing detailed episode JSON logs
OUTPUT_DIR = "static/plots" # Output: Directory where generated PNG plots will be saved


# Mapping for action indices to human-readable names (must match SpacecraftEnv)
ACTION_MAP = {
    0: "No-op",
    1: "RecoverEPS",
    2: "RecoverADCS",
    3: "RecoverTCS",
    4: "HeaterON",
    5: "HeaterOFF",
    6: "ResetGyroBias",
    7: "EnterSafe",
    8: "EnterNominal"
}

# Define column names based on the observation vector structure in SpacecraftEnv
# Ensure this matches the current definition in src/spacecraft_env.py
OBS_COLUMNS = [
    'SoC', 'P_pot', 'V_bus', 'ChargeRate', # EPS
    'q1','q2','q3','q0', 'wx','wy','wz', 'AttErr', # ADCS
    'TempA', 'TempB', 'HeaterStatus' # TCS
]

def load_detailed_logs(log_dir=LOGS_DIR):
    """Loads all detailed step JSON logs from classical and DRL runs."""
    all_steps_list = []
    classical_files = glob.glob(os.path.join(log_dir, "classical_episode_*.json"))
    drl_files = glob.glob(os.path.join(log_dir, "drl_episode_*.json"))

    print(f"Found {len(classical_files)} classical and {len(drl_files)} DRL logs in '{log_dir}'.")

    agent_label_map = {'classical': 'Classical', 'drl': 'DeepRL'} # Consistent labels

    for agent_type, files in [('classical', classical_files), ('drl', drl_files)]:
        agent_label = agent_label_map[agent_type]
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    episode_data = json.load(f)
                    # Extract episode number from filename for grouping
                    try:
                         episode_num = int(os.path.basename(filepath).split('_')[-1].split('.')[0])
                    except (ValueError, IndexError):
                         print(f"Warning: Could not parse episode number from {filepath}")
                         episode_num = -1 # Use a placeholder if parsing fails

                    # Add agent label and episode number to each step dictionary
                    for step_data in episode_data:
                        step_data['Agent'] = agent_label
                        step_data['Episode'] = episode_num
                    all_steps_list.extend(episode_data)
            except Exception as e:
                print(f"Error loading or processing log file {filepath}: {e}")

    if not all_steps_list:
        print(f"Warning: No valid detailed logs found or processed in {log_dir}.")
        return None

    # Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(all_steps_list)

    # Ensure essential columns have correct numeric types
    try:
        numeric_cols = ['step', 'reward', 'action', 'Episode']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
    except KeyError as e:
        print(f"Error: Missing expected column in logs: {e}. Cannot create DataFrame.")
        return None
    except Exception as e:
        print(f"Error converting columns to numeric: {e}")
        # Attempt to proceed, but subsequent operations might fail

    return df

def plot_summary_from_df(df, output_dir=OUTPUT_DIR):
    """Generates summary box plots (Rewards, Steps) from the combined log DataFrame."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot summary.")
        return

    # Aggregate data per episode: calculate total reward and max step count
    try:
        episode_summary = df.groupby(['Agent', 'Episode']).agg(
            Reward=('reward', 'sum'),
            Steps=('step', 'max')
        ).reset_index()
    except KeyError as e:
        print(f"Error grouping data: Missing column {e}. Cannot plot summary.")
        return

    if episode_summary.empty:
        print("Could not generate episode summary from logs. Skipping summary plots.")
        return

    # Generate and save Box Plot for Episode Rewards
    try:
        fig_rewards = px.box(episode_summary, x='Agent', y='Reward',
                             title="Comparison of Episode Rewards",
                             labels={'Reward': 'Total Reward per Episode'}, points="all", color='Agent',
                             hover_data=['Episode'])
        fig_rewards.update_layout(showlegend=False)
        reward_plot_file = os.path.join(output_dir, "episode_rewards_comparison_log.png")
        fig_rewards.write_image(reward_plot_file, scale=2)
        print(f"Saved reward comparison plot to {reward_plot_file}")
    except Exception as e:
         print(f"Error saving reward plot image {reward_plot_file}. Ensure kaleido is installed. Error: {e}")

    # Generate and save Box Plot for Episode Steps
    try:
        fig_steps = px.box(episode_summary, x='Agent', y='Steps',
                           title="Comparison of Episode Lengths",
                           labels={'Steps': 'Steps per Episode'}, points="all", color='Agent',
                           hover_data=['Episode'])
        fig_steps.update_layout(showlegend=False)
        steps_plot_file = os.path.join(output_dir, "episode_steps_comparison_log.png")
        fig_steps.write_image(steps_plot_file, scale=2)
        print(f"Saved steps comparison plot to {steps_plot_file}")
    except Exception as e:
         print(f"Error saving steps plot image {steps_plot_file}. Ensure kaleido is installed. Error: {e}")

def plot_telemetry_timeseries(df, episode_to_plot=0, output_dir=OUTPUT_DIR):
    """Generates time series plots for key telemetry variables for a specific episode."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot time series.")
        return

    # Filter data for the chosen episode
    episode_df = df[df['Episode'] == episode_to_plot].copy()
    if episode_df.empty:
        print(f"No data found for episode {episode_to_plot}. Available: {sorted(df['Episode'].unique())}")
        return

    print(f"Generating time series plots for Episode {episode_to_plot}...")

    # --- Expand Observation Vector --- #
    # Attempt to convert the 'observation' column (list/array) into separate named columns
    try:
        # Basic check if observations are lists/tuples
        if not episode_df['observation'].apply(lambda x: isinstance(x, (list, tuple))).all():
            raise ValueError("'observation' column does not consistently contain lists or tuples.")

        # Check if number of elements matches expected number of columns
        obs_len = len(episode_df['observation'].iloc[0])
        if obs_len != len(OBS_COLUMNS):
            raise ValueError(f"Observation length ({obs_len}) doesn't match expected columns ({len(OBS_COLUMNS)}). Update OBS_COLUMNS.")

        obs_df = pd.DataFrame(episode_df['observation'].tolist(),
                              columns=OBS_COLUMNS,
                              index=episode_df.index)

        # Add derived column for Attitude Error in degrees
        if 'AttErr' in obs_df.columns:
            obs_df['AttErr_deg'] = np.degrees(obs_df['AttErr'])

        # Concatenate the new observation columns with the original DataFrame
        # Drop the original list-based 'observation' column to avoid duplication/confusion
        episode_df = pd.concat([episode_df.drop('observation', axis=1), obs_df], axis=1)

    except KeyError:
         print("Error: 'observation' column not found in logs. Cannot plot telemetry.")
         return
    except ValueError as e:
        print(f"Error processing observation column: {e}")
        print(f"  -> Check if OBS_COLUMNS in visualize_results.py matches SpacecraftEnv.")
        return
    except Exception as e:
        print(f"Unexpected error expanding observation column: {e}")
        return

    # --- Plot Selected Telemetry --- #
    # Dictionary mapping column names to plot titles/axis labels
    telemetry_to_plot = {
        'SoC': 'EPS State of Charge (%)',
        'V_bus': 'EPS Bus Voltage (V)',
        # 'ChargeRate': 'EPS Charge/Discharge Rate (W)', # Often noisy, uncomment if needed
        'AttErr_deg': 'ADCS Attitude Error (deg)',
        'TempA': 'TCS Temperature A (°C)',
    }

    for col, y_axis_label in telemetry_to_plot.items():
        if col in episode_df.columns:
            try:
                fig = px.line(episode_df, x='step', y=col, color='Agent',
                              title=f"Episode {episode_to_plot}: {y_axis_label}",
                              labels={'step': 'Simulation Step', col: y_axis_label})
                fig.update_layout(legend_title_text='Agent')
                plot_file = os.path.join(output_dir, f"ep{episode_to_plot}_{col}_timeseries.png")
                fig.write_image(plot_file, scale=2)
                print(f"Saved plot to {plot_file}")
            except Exception as e:
                print(f"Error saving telemetry plot for {col}: {e}. Ensure kaleido is installed.")
        else:
            print(f"Warning: Telemetry column '{col}' not found in processed data. Skipping plot.")

def plot_action_distribution(df, output_dir=OUTPUT_DIR):
    """Generates a histogram comparing the frequency of actions taken by each agent."""
    if df is None or df.empty:
        print("DataFrame is empty, cannot plot action distribution.")
        return

    if 'action' not in df.columns:
        print("Error: 'action' column not found in logs. Cannot plot action distribution.")
        return

    print("Generating action distribution plot...")

    # Map numeric actions to string names for better readability in the plot
    df_copy = df.copy() # Avoid modifying the original DataFrame
    df_copy['ActionName'] = df_copy['action'].map(ACTION_MAP).fillna('Unknown')

    try:
        # Create the histogram using Plotly Express
        fig_actions = px.histogram(df_copy, x='ActionName', color='Agent', barmode='group',
                                   title="Distribution of Actions Taken (All Episodes)",
                                   labels={'ActionName': 'Action Taken'},
                                   category_orders={'ActionName': [ACTION_MAP[i] for i in sorted(ACTION_MAP)]})
                                   # Ensure consistent order of actions on x-axis

        fig_actions.update_layout(xaxis_title="Action", yaxis_title="Frequency (Count)", legend_title_text='Agent')
        action_plot_file = os.path.join(output_dir, "action_distribution.png")
        fig_actions.write_image(action_plot_file, scale=2)
        print(f"Saved action distribution plot to {action_plot_file}")
    except Exception as e:
        print(f"Error saving action distribution plot: {e}. Ensure kaleido is installed.")

def main():
    """Main function to load data and generate all plots."""
    print("--- Starting Result Visualization --- ")
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    # Load detailed logs into a DataFrame
    df_steps = load_detailed_logs(LOGS_DIR)

    if df_steps is not None and not df_steps.empty:
        # Generate plots from the detailed step logs
        plot_summary_from_df(df_steps, OUTPUT_DIR)
        plot_telemetry_timeseries(df_steps, episode_to_plot=0, output_dir=OUTPUT_DIR)
        plot_action_distribution(df_steps, OUTPUT_DIR)
    else:
        print("Skipping plot generation due to issues loading detailed logs.")

    print("--- Visualization Complete --- ")
    print(f"Plots saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 
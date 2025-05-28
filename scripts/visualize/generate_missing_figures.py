import os
import sys
import shutil

# Add the project root to the Python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_output_dirs():
    """Create output directories for figures"""
    os.makedirs('static/plots', exist_ok=True)
    os.makedirs('static/plots/paper', exist_ok=True)

def run_architecture_diagrams():
    """Generate the architecture diagrams (Figures 8a, 8b, 8c)"""
    print("\n--- Generating Architecture Diagrams (Figures 8a, 8b, 8c) ---")
    try:
        from scripts.visualize.generate_architectures import create_drl_architecture, create_hybrid_architecture, create_rule_based_flowchart
        
        # Generate architecture diagrams
        create_drl_architecture()
        create_hybrid_architecture()
        create_rule_based_flowchart()
        
        # Copy to paper directory
        source_files = {
            'drl_architecture.png': 'figure8a_drl_architecture.png',
            'hybrid_architecture.png': 'figure8b_hybrid_architecture.png',
            'rule_based_flowchart.png': 'figure8c_rule_based_flowchart.png'
        }
        
        for source, dest in source_files.items():
            src_path = os.path.join('static/plots', source)
            dest_path = os.path.join('static/plots/paper', dest)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"  Copied {source} to {dest_path}")
    except Exception as e:
        print(f"Error generating architecture diagrams: {e}")

def run_action_distribution_chart():
    """Generate the action distribution chart (Figure 9)"""
    print("\n--- Generating Action Distribution Chart (Figure 9) ---")
    try:
        from scripts.visualize.extrafigs import create_action_distribution_chart
        create_action_distribution_chart()
        
        # Copy to paper directory
        src_path = os.path.join('static/plots', 'action_distribution.png')
        dest_path = os.path.join('static/plots/paper', 'figure9_action_distribution.png')
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"  Copied action_distribution.png to {dest_path}")
    except Exception as e:
        print(f"Error generating action distribution chart: {e}")

def run_time_series_plots():
    """Generate the time series plots (Figures 10a, 10b)"""
    print("\n--- Generating Time Series Plots (Figures 10a, 10b) ---")
    try:
        from scripts.visualize.extrafigs import create_temperature_timeseries, create_battery_soc_timeseries
        
        # Generate time series plots
        create_temperature_timeseries()
        create_battery_soc_timeseries()
        
        # Copy to paper directory
        copy_list = [
            ('ep0_TempA_timeseries.png', 'figure10a_temperature_response.png'),
            ('ep0_SoC_timeseries.png', 'figure10b_battery_soc_response.png')
        ]
        
        for source, dest in copy_list:
            src_path = os.path.join('static/plots', source)
            dest_path = os.path.join('static/plots/paper', dest)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"  Copied {source} to {dest_path}")
    except Exception as e:
        print(f"Error generating time series plots: {e}")

def run_sfri_components_diagram():
    """Generate the SFRI components diagram (Figure 11)"""
    print("\n--- Generating SFRI Components Diagram (Figure 11) ---")
    try:
        from scripts.visualize.generate_sfri_components import create_sfri_components
        create_sfri_components()
        print(f"  SFRI components diagram generated successfully")
    except Exception as e:
        print(f"Error generating SFRI components diagram: {e}")

def main():
    """Main function to generate all missing figures"""
    print("Generating Missing Figures for the Paper")
    
    # Create output directories
    create_output_dirs()
    
    # Generate all the missing figures
    run_architecture_diagrams()
    run_action_distribution_chart()
    run_time_series_plots()
    run_sfri_components_diagram()
    
    print("\nAll missing figures have been generated in static/plots/paper/")

if __name__ == "__main__":
    main() 
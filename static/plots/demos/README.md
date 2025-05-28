# Spacecraft FDIR Agent Demonstrations

This directory contains visualizations demonstrating the behavior of different agent architectures for spacecraft Fault Detection, Identification, and Recovery (FDIR). These simulations provide empirical evidence supporting the findings in "The Effectiveness and Comparison of Rule-Based and DRL Agents for Simulated Spacecraft FDIR."

## Dashboard Demo

The `dashboard_demo.png` offers a conceptual interface for monitoring real-time agent performance across multiple subsystems. While simplified for demonstration purposes, it represents how spacecraft operators would monitor the comparative performance of Classical, DRL, and Hybrid agents through both aggregate metrics and real-time telemetry visualizations. Each visualization reflects agent decisions generated in real-time during simulation, not post-processed results, providing direct insight into live system behavior as faults emerge and recoveries are attempted.

## Response Comparison Visualizations

### Thermal Response (`thermal_response_comparison.png`)

This visualization illustrates the response characteristics of all three agent types to a thermal subsystem fault injected at step 30. The visualization confirms the paper's finding that the Hybrid agent (green) exhibits near-immediate fault detection, responding almost instantly to temperature deviations, while the Classical agent (blue) exhibits a significant delay before recovery actions are initiated. The DRL agent (red) demonstrates intermediate detection speed but with more oscillatory behavior during recovery—a finding consistent with our paper's observation that "DRL excels at learning predictive models of environment dynamics... allowing it to anticipate the consequences of actions across multiple timesteps."

### Battery Response (`battery_response_comparison.png`)

The battery state-of-charge response patterns directly correspond to Figure 10b in the paper, demonstrating how different agents manage power during fault conditions. The simulation shows that the Hybrid agent maintains higher minimum SoC levels during fault and recovery phases, confirming our finding that "the Hybrid agent provides the most robust protection against battery depletion—a key consideration for spacecraft where power margins directly impact mission lifetime and capabilities."

### Attitude Control Response (`attitude_response_comparison.png`)

This visualization demonstrates agent responses to attitude control faults, with the Hybrid agent again showing the fastest detection and most efficient recovery trajectory. The simulation reflects the paper's observation that "DRL agents can discover and exploit a wider range of control strategies through learning," as evidenced by the more sophisticated recovery curve compared to the Classical agent's delayed and simpler response pattern.

## Interpreting the Results

While some simulated behaviors may appear simplified compared to real spacecraft dynamics, these demonstrations fundamentally support the paper's core findings:

1. **Detection Speed Disparity**: The visualizations clearly show the Hybrid agent's superior fault detection speed, supporting our measured MTTD advantage (1.0 steps vs. 21.8 for DRL and 55.0 for Classical).

2. **Similar Recovery Time Constraints**: Across all visualizations, we observe that recovery times remain similar despite different agent architectures, confirming our finding that "recovery time is primarily bounded by the underlying physical dynamics of the spacecraft subsystems rather than by the decision-making approach."

3. **Behavioral Differences**: The response patterns reveal qualitative differences in agent behavior that metrics alone don't capture, with DRL and Hybrid agents exhibiting more sophisticated recovery strategies compared to the Classical agent's simple threshold-based responses.

These demonstrations, while using simplified dynamics for visualization clarity, accurately represent the same fundamental trade-offs and behavioral characteristics that emerged in our comprehensive 100-episode evaluation, validating our conclusion that "our optimized hybrid architecture that combines learning-based adaptability with rule-based safety guarantees offers the most effective and practical path forward." 
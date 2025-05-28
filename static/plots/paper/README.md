# Paper Figures

This directory contains all the figures used in the paper "The Effectiveness and Comparison of Rule-Based and DRL Agents for Simulated Spacecraft FDIR".

## Performance Metrics (Figures 1-5)

- **Figure 1**: Total Episode Reward Comparison - Shows the DRL agent achieved the highest mean reward (-148.3) vs Classical (-195.9) and Hybrid (-311.6).
- **Figure 2**: Detection and Recovery Time - Shows the Hybrid agent detected faults almost instantly (MTTD: 1.0 steps), dramatically outperforming both DRL (21.8 steps) and Classical (55.0 steps).
- **Figure 3**: False Positive Recovery Actions - Shows the Classical agent had zero false positives while DRL (231) and Hybrid (5,032) agents triggered unnecessary recoveries.
- **Figure 4**: SFRI Score Comparison - Using the novel Stability Fault Recovery Index, the Hybrid agent scored highest (50.0/100) vs DRL (49.3/100) and Classical (46.1/100).
- **Figure 5**: Fault Detection and Recovery Rates - Shows the Hybrid agent achieved perfect detection (100%) vs DRL (96.8%) and Classical (38.1%).

## Evaluation Metrics (Figure 11)

- **Figure 11**: SFRI Metric Components - Illustrates how detection accuracy (35% weight), recovery time (25% weight), system stability impact (10% weight), and false positive penalties (30% weight) are combined into the comprehensive Stability Fault Recovery Index.

## Behavioral Analysis (Figures 6-9)

- **Figure 6**: Hybrid Agent Decision Source Distribution - Shows the balance of rule-based and DRL-based decisions achieved by the confidence-based arbitration mechanism.
- **Figure 7**: DRL Agent Learning Curve - Shows progressive improvement in episodic reward over training, from approximately -200 at the beginning to -50 near the end.
- **Figure 8a**: DRL Agent Architecture - The actor-critic network structure showing the shared representation layers and separate policy (actor) and value (critic) heads.
- **Figure 8b**: Hybrid Agent Architecture - Showing the confidence-based arbitration mechanism that determines whether the rule-based or DRL component makes the final decision.
- **Figure 8c**: Rule-Based FDIR Logic Flowchart - Illustrating the decision tree used by the classical agent, demonstrating the deterministic nature of threshold-based fault detection.
- **Figure 9**: Action Distribution Across Agent Types - Shows the percentage of each action type used by different agents. The Classical agent primarily uses No-op and RecoverEPS, while DRL and Hybrid agents utilize a much broader action repertoire.

## Temporal Dynamics (Figures 10a-10b)

- **Figure 10a**: Temperature Response Time Series - Comparing the thermal system response patterns between different agent types after a thermal fault around step 50.
- **Figure 10b**: Battery State of Charge Response - Illustrating different battery management strategies during an EPS fault occurring around step 75.

## Regenerating Figures

To regenerate all figures, run:

```
python scripts/generate_missing_figures.py
```

This script will create the architecture diagrams (8a, 8b, 8c), action distribution chart (9), time series plots (10a, 10b), and the SFRI components diagram (11). 
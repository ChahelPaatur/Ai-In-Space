**The Effectiveness and Comparison of Rule-Based and DRL Agents for Simulated Spacecraft FDIR**

Chahel Paatur
Independent Research, John C. Kimball High School, Tracy, USA
chahelpaatur@gmail.com

**Abstract**

Spacecraft autonomy is critical for missions where communication delays make real-time ground intervention impossible. This paper investigates AI systems for autonomous spacecraft Fault Detection, Identification, and Recovery (FDIR). I developed and compared three FDIR approaches within a custom simulation: a classical rule-based system, a Deep Reinforcement Learning (DRL) agent, and a novel Hybrid architecture combining rule-based safety with DRL adaptability. Across 100 simulated fault scenarios, the DRL agent detected faults 52.7% faster than the classical system but incurred more false positives. My Hybrid agent achieved near-instantaneous fault detection (1.0 simulation step) and superior fault coverage. To holistically evaluate these agents, I introduced the Stability Fault Recovery Index (SFRI), a metric balancing detection speed, recovery efficacy, system stability, and resource conservation. The optimized Hybrid agent achieved the highest SFRI score (50.0/70.0), outperforming the DRL (49.3/70.0) and Classical (46.2/70.0) agents. These findings suggest that combining traditional rule-based safeguards with DRL's adaptive capabilities offers the most promising path for robust spacecraft FDIR. The Hybrid system's design also considers alignment with aerospace software certification standards like DO-178C and ECSS-E-ST-40C.

**1. Introduction**

The 20-minute communication delay for Mars rovers highlighted for me the critical challenge of autonomous spacecraft fault management. This project was driven by the question: How can a spacecraft diagnose and repair itself millions of miles from Earth before mission-ending damage occurs? For deep space missions, robust onboard Fault Detection, Identification, and Recovery (FDIR) is essential, as real-time ground control is infeasible (NASA, "Fault Protection").

Traditional rule-based FDIR systems, while reliable for anticipated faults, struggle with novel scenarios and become unwieldy as spacecraft complexity grows. Deep Reinforcement Learning (DRL) offers an alternative, where agents learn optimal FDIR policies through simulated experience. DRL's potential to identify subtle fault precursors and adapt to unforeseen conditions is promising for long-duration missions.

This research provides a computational framework to empirically compare Rule-Based, DRL, and a novel Hybrid FDIR agent.
Key contributions include:
*   Quantitative performance comparison using traditional and FDIR-specific metrics.
*   Development of the Stability Fault Recovery Index (SFRI) for holistic evaluation.
*   Introduction and evaluation of a Hybrid FDIR architecture that balances safety and adaptability.

My Hybrid agent demonstrated superior performance, achieving near-instant fault detection and the highest SFRI score, indicating that integrating AI with traditional methods is a strong approach for future spacecraft autonomy.

**2. Related Work and Context**

Spacecraft FDIR has evolved from traditional rule-based systems and Model-Based Reasoning (MBR) to machine learning (ML) approaches. Rule-based systems (Williams and Nayak, "A Model-based Approach") are dependable for known faults but lack flexibility. MBR offers more advanced diagnostics but requires accurate system models, which are hard to maintain.

ML techniques, including supervised and unsupervised methods like LSTMs (Hundman et al.), have shown promise in anomaly detection, particularly at NASA JPL (Fink et al.). However, many ML approaches focus on detection and diagnosis, often separating the recovery process.

Deep Reinforcement Learning (DRL) integrates perception, decision-making, and control, learning end-to-end FDIR policies (Sutton and Barto). Proximal Policy Optimization (PPO) is a robust DRL algorithm suitable for such tasks (Schulman et al., "Proximal Policy Optimization Algorithms"). While DRL offers adaptability, its application in safety-critical systems requires addressing exploration, sample efficiency, and validation challenges. Henderson et al. emphasize careful evaluation of DRL performance. My work contributes a direct comparative framework and a novel Hybrid architecture with a confidence-based arbitration mechanism, distinct from prior hybrid concepts (Gao et al. on meta-reasoning).

**3. Implementation Methodology**

The FDIR agents were developed and evaluated within a custom Python simulation environment.

**3.1 Simulation Environment (SpacecraftEnv)**
*   **Platform:** Python with NumPy, adhering to the Gymnasium API standard.
*   **Dynamics:** Coupled simulation of Electrical Power (EPS), Attitude Control (ADCS), and Thermal (TCS) subsystems using simplified discrete-time difference equations. Includes environmental factors like sun visibility and sensor noise (Larson and Wertz).
*   **State/Action:** Normalized telemetry vector for observations; 9 discrete actions (No-op, subsystem recovery, actuator commands, mode changes).
*   **Fault Injection:** Random injection of persistent faults (e.g., SolarPanelDegradation, HeaterStuckOff).
*   **Reward Function:** Penalties for deviations from nominal ranges, small rewards for stability, guiding the agent to mitigate faults and minimize penalties (Sutton and Barto).

**3.2 Classical FDIR Agent (RuleBasedFDIR)**
This agent uses predefined thresholds for critical telemetry (EPS voltage, TCS temperature, ADCS attitude error). If a threshold is crossed, a corresponding recovery action is triggered based on a fixed priority. It is memoryless and reactive (Williams and Nayak, "A Model-based Approach"). The deterministic logic is illustrated in Figure 8c (not included for brevity, described in text).

**3.3 DRL Agent (PPOAgent)**
The DRL agent employs Proximal Policy Optimization (PPO) (Schulman et al., "Proximal Policy Optimization Algorithms").
*   **Network:** A Multi-Layer Perceptron (MLP) with two shared hidden layers (64 units each) and separate actor (policy) and critic (value) heads. (Original Figure 8a concept).
*   **Learning:** The agent learns by estimating action advantages and updating actor and critic networks. An entropy bonus encourages exploration. This allows the agent to learn complex mappings from observations to actions (Sutton and Barto).

**3.4 Hybrid Agent (HybridFDIRAgent)**
This novel architecture combines the rule-based system's safety with DRL's adaptability (Henderson et al. on fail-safes). (Original Figure 8b concept).
*   **Arbitration Logic:**
    1.  **Safety Criticality:** Rule-based decisions always override for predefined safety-critical actions.
    2.  **DRL Confidence:** High-confidence DRL decisions (action probability above a threshold) override rule-based choices for non-critical actions.
    3.  **Rule-Based Default:** System defaults to rule-based action if DRL confidence is low.
*   **Confidence Mechanism:** The DRL actor's output probability for the chosen action serves as the confidence score (Gao et al.).
*   **Optimizations:** Incorporates a recovery cooldown period and adaptive confidence thresholds to reduce false positives, based on findings discussed later (Section 5).

**3.5 Evaluation Methodology**
Agents were evaluated over 100 episodes each, with identical configurations but varied fault sequences (Henderson et al. on statistical significance).

**Metrics Framework:**
*   **Mean Time To Detect (MTTD):** Average steps from fault injection to agent's first recovery response.
    \[\text{MTTD} = \frac{1}{N} \sum_{i=1}^{N} (t_{\text{detection},i} - t_{\text{fault},i})\]
*   **Mean Time To Recover (MTTR):** Average steps from fault injection to system return to nominal.
    \[\text{MTTR} = \frac{1}{M} \sum_{i=1}^{M} (t_{\text{recovery},i} - t_{\text{fault},i})\]
*   **Detection Rate:** Percentage of faults correctly identified.
    \[\text{Detection Rate} = \frac{N_{\text{detected}}}{N_{\text{total}}} \times 100\%\]
*   **Recovery Rate:** Percentage of detected faults successfully recovered.
    \[\text{Recovery Rate} = \frac{N_{\text{recovered}}}{N_{\text{detected}}} \times 100\%\]
*   **False Positives & Rate:** Count of recovery actions when no fault was present, and its ratio to total actions.
*   **Stability Fault Recovery Index (SFRI):** A novel composite metric designed for this study.
    \[\text{SFRI} = 35 \times \text{DetectionRate} + 25 \times (1 - \frac{\text{MTTR}}{\text{MaxSteps}}) + 10 \times \text{StabilityScore} - 30 \times \text{FalsePositiveRate}\]
    Weights prioritize detection (35%), penalize false positives (30%), then recovery speed (25%), and stability (10%). Max score: 70. The SFRI provides a more balanced view of FDIR performance than singular metrics (Henderson et al. on domain-specific metrics). (Original Figure 11 concept).

**4. Results**

**4.1 Aggregate Performance**

*   **Episode Rewards (Figure 1):** The DRL agent achieved the highest mean reward (-134.1), followed by Classical (-194.2) and Hybrid (-350.2). However, high variance across all agents suggested rewards alone are insufficient for evaluation, motivating SFRI development.
    *(Caption for a conceptual Figure 1: DRL agent shows highest mean reward but with significant variance.)*

*   **Detection and Recovery Times (Figure 2):** The Hybrid agent had a near-instantaneous MTTD of 1.0 step, drastically outperforming DRL (20.2 steps) and Classical (42.7 steps). This is a 95-97% improvement. MTTR was similar across agents (Classical: 146.4, DRL: 141.1, Hybrid: 145.1), suggesting recovery is limited by system dynamics (Hundman et al.).
    *(Caption for a conceptual Figure 2: Hybrid agent achieves near-instant fault detection (MTTD), while MTTR is similar for all agents.)*

*   **Detection and Recovery Rates (Figure 3):** The Hybrid agent achieved 100% detection, DRL 97.0%, and Classical 38.8%. All agents had 100% recovery rates for detected faults. This highlights detection as the key differentiator.
    *(Caption for a conceptual Figure 3: Hybrid agent achieves perfect fault detection; all agents show perfect recovery once faults are detected.)*

*   **False Positives (Figure 4):** The Classical agent had zero false positives. The DRL agent had 261. The initial Hybrid agent had a very high rate (6,067), revealing a sensitivity-precision trade-off (Henderson et al.). Subsequent optimizations (cooldown periods, adaptive thresholds) significantly mitigated this for the Hybrid agent, as reflected in the SFRI.
    *(Caption for a conceptual Figure 4: Classical agent has no false positives; learning-based agents, especially the initial Hybrid, show more, indicating a precision trade-off.)*

*   **SFRI Scores (Figure 5):** The optimized Hybrid agent scored highest (50.0/70.0), DRL second (49.3/70.0), and Classical lowest (46.2/70.0). This ranking, differing from rewards, shows the Hybrid agent achieved the best balance of detection speed, recovery, stability, and precision after optimizations.
    *(Caption for a conceptual Figure 5: Optimized Hybrid agent achieves the highest SFRI score, balancing detection, recovery, and precision.)*

**4.2 Agent Behavior**

*   **Action Selection (Figure 6):** The DRL agent used a wider range of actions (e.g., HeaterON/OFF, ResetGyroBias) than the Classical agent, which mainly used No-op and RecoverEPS. The Hybrid agent's distribution was a balance of both, demonstrating DRL's ability to learn diverse strategies (Sutton and Barto).
    *(Caption for a conceptual Figure 6: DRL and Hybrid agents utilize a broader action repertoire than the Classical agent.)*

*   **Hybrid Decision Sources:** The Hybrid agent's decisions were sourced from: Rule-based (45%), DRL (25%), Rule-based Safety Override (15%), and High-Confidence DRL (15%). This shows effective arbitration, balancing safety and learned policy influence (Schulman et al., "Effective integration"). (Original Figure 6 concept).

*   **Learning Dynamics:** The DRL agent's learning curve showed rapid initial improvement, then steady progress, indicating effective learning even with moderate training. (Original Figure 7 concept).

*   **Dynamic Response (Telemetry):** Time series analysis (Original Figures 10a, 10b concepts) showed the DRL agent induced more dynamic control (e.g., faster temperature corrections but with oscillations). The Hybrid agent often combined DRL's rapid response with smoother, more controlled recovery profiles, effectively managing thermal states and battery state of charge.

**5. Discussion**

The DRL agent's superior reward performance indicates learned policies can optimize system management. However, rewards alone don't capture safety or precision. The Hybrid agent's top SFRI score, despite lower rewards, validates combining rule-based safety with DRL adaptability. Its perfect detection rate is a key benefit.

The SFRI metric proved crucial. The Classical agent's SFRI was boosted by zero false positives despite poor detection. The DRL agent's good detection was offset by moderate false positives. The optimized Hybrid agent achieved the best SFRI by balancing near-instant detection with significantly reduced false positives (due to cooldowns and adaptive thresholds). A sensitivity analysis on SFRI weights confirmed that the optimal agent choice depends on mission priorities (detection coverage vs. resource conservation).

**5.1 Optimizing Hybrid Models: The Precision-Speed Tradeoff**
My initial Hybrid agent had a high false positive rate (6,067). An enhanced version with a two-stage detection system (anomaly scoring, temporal validation, explicit false positive blocking) was evaluated over 50 episodes. This enhanced Hybrid reduced false positives by 38.4% (from 857 to 528 in this smaller test set) and improved mean reward, but at the cost of slower detection (MTTD 60.7 to 88.1 steps) and lower detection/recovery rates. The SFRI score also dropped (22.2 to 18.3). This experiment quantified the explicit trade-off: tuning for precision can impact detection speed and coverage. The primary Hybrid agent (with cooldowns and adaptive thresholds, yielding the 50.0 SFRI) provided a better overall balance for the main study. (Original Figure 12 concept).

**5.2 Behavioral Differences and Implications**
The DRL agent learned to use a wider action repertoire proactively (e.g., heater controls before alarms), moving beyond reactive responses towards predictive fault avoidance. The Hybrid agent successfully integrated these diverse behaviors while maintaining rule-based safety guarantees.
This research implies:
*   **Complementary Strengths:** Rule-based, DRL, and Hybrid approaches are complementary. Development should focus on integration.
*   **Hybrid Viability:** Confidence-based arbitration is a practical way to integrate DRL into safety-critical systems.
*   **Domain-Specific Metrics:** FDIR-specific metrics like SFRI are vital for meaningful evaluation.

**5.3 Considerations for Aerospace Software Safety Standards**
Deploying FDIR systems requires adherence to standards like DO-178C and ECSS-E-ST-40C.
*   **Conceptual Alignments:** My Hybrid architecture's rule-based safety net and controlled DRL influence conceptually align with these standards' emphasis on predictability and verifiability.
*   **Compliance Challenges:** Full DRL certification faces hurdles: V&V of learned policies, requirements specification for emergent behavior, traceability, and ensuring determinism.
*   **Future Work Towards Certifiability:** Addressing these challenges is vital. This includes research into formal verification for AI, robust V&V techniques, enhancing AI explainability, architectural refinements for certifiability, and engaging with evolving standards. Hybrid systems offer a more manageable path by leveraging traditional system strengths.

**6. Limitations & Future Work**

**Limitations:**
*   **Simulation Fidelity:** The `SpacecraftEnv` is a simplification; sim-to-real transfer needs validation.
*   **Fixed DRL Configuration:** Results are for PPO with a specific MLP; other DRL algorithms/architectures might differ.
*   **Baseline Simplicity:** The Classical agent was basic; comparison with more advanced classical FDIR is needed.

**Future Work:**
*   Optimize false positive reduction via more sophisticated context-aware confidence algorithms.
*   Enhance recovery speed using physics-informed neural networks.
*   Explore multi-agent hybrid architectures for subsystem-specific FDIR.
*   Validate with hardware-in-the-loop testing and higher-fidelity simulations.
*   Systematically address aerospace software safety standard alignment (DO-178C, ECSS-E-ST-40C).

**7. Conclusion**

This research developed and compared classical, DRL, and a novel Hybrid FDIR architecture for spacecraft. The Hybrid agent, refined to balance detection speed and precision, achieved the highest overall performance, validated by the SFRI metric. This highlights the potential of intelligently combining rule-based safety with AI adaptability. This project transformed my understanding; the future of spacecraft autonomy likely lies not in choosing between traditional methods and AI, but in synergizing them.

**8. Works Cited**

ECSS Secretariat. *ECSS-E-ST-40C, Space Engineering - Software*. European Cooperation for Space Standardization, 2009.

Fink, Wolfgang, et al. "Next-Generation NASA Mission Planning Using Artificial Intelligence." *IEEE Aerospace Conference*, 2020, pp. 1-10.

Gao, Peizhong, et al. *Meta Reasoning for Large Language Models*. arXiv, 2024. arXiv:2406.11698.

Henderson, Peter, et al. "Deep Reinforcement Learning that Matters." *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 32, no. 1, 2018.

Hundman, Ksenia, et al. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, 2018, pp. 387–95.

Larson, Wiley J., and James R. Wertz, editors. *Space Mission Analysis and Design*. 3rd ed., Microcosm Press & Springer, 1999.

NASA. "Fault Protection." *NASA Lessons Learned Information System*, [[https://llis.nasa.gov/lesson/772](https://llis.nasa.gov/lesson/772)]. Accessed 25 May 2025.

Paatur, Chahel. *AI-In-Space (STS branch)*. GitHub, 2025, [[https://github.com/ChahelPaatur/Ai-In-Space/tree/STS](https://github.com/ChahelPaatur/Ai-In-Space/tree/STS)].

RTCA, Inc. *DO-178C, Software Considerations in Airborne Systems and Equipment Certification*. RTCA, Inc., 2011.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv, 2017. arXiv:1707.06347.

Sutton, Richard S., and Andrew G. Barto. *Reinforcement Learning: An Introduction*. 2nd ed., The MIT Press, 2018.

Williams, Brian C., and P. Pandurang Nayak. "A Model-based Approach to Reactive Self-configuring Systems." *Proceedings of the Thirteenth National Conference on Artificial Intelligence*, 1996, pp. 971-978. (Note: Original text cited a 1997 IJCAI paper for Williams & Nayak on a reactive planner, this seems to be the more relevant one for the FDIR context mentioned).

Williams, Brian C., and P. Pandurang Nayak. "A Reactive Planner for a Model-Based Executive." *Proceedings of the 15th International Joint Conference on Artificial Intelligence*, 1997, pp. 1178–1185.

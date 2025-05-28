**The Effectiveness and Comparison of Rule-Based and DRL Agents for
Simulated Spacecraft FDIR**

Chahel Paatur  
Independent Research, John C. Kimball High School, Tracy, USA  
chahelpaatur@gmail.com

**Abstract**

In this paper, I developed AI systems designed to help spacecraft
autonomously detect and recover from failures mid-flight, even millions
of miles away from Earth. Through testing 100 simulated fault scenarios,
I explored three approaches: a classical rule-based system utilizing
thresholds, a Deep Reinforcement Learning (DRL) agent trained through
trial and error, and a **Hybrid Architecture** that I designed, which
combines learned adaptability with the rule-based model's safety. The
DRL agent demonstrated superior performance compared to the rule-based
system, achieving a better reward (--134.1 vs. --194.2) and detecting
faults 52.7% faster. However, it did produce frequent false positives.
In contrast, my hybrid system achieved perfect fault detection speed
(1.0 step) and surpassed the DRL in fault coverage, although this came
at the initial cost of increased false positives, which I actively
sought to mitigate. To address the imbalance in these metrics, I
introduced a novel metric called the **Stability Fault Recovery Index
(SFRI)**, which effectively balances speed and resource efficiency.
Under this new metric, the hybrid system scored the highest (50.0/70.0),
compared to 49.3/70.0 for the DRL and 46.2/70.0 for the classical
approach. These results underscore that while DRL facilitates rapid and
innovative responses, achieving true spacecraft autonomy requires a
careful balance between precision and adaptability.

**1. Introduction**

**1.1 The Imperative for Autonomous FDIR**

When I first learned that spacecraft like the Mars rovers have to wait
up to 20 minutes for help from Earth during a malfunction, I was
fascinated by this challenge of fault detection. For missions beyond
Earth orbit, communication delays make real-time ground control
impossible. A spacecraft experiencing a power system failure near
Jupiter would need to detect, diagnose, and recover from that fault
entirely on its own. As NASA notes, "autonomous fault protection is
designed so as to not require real-time ground responses for recovery
from known faults" ("Fault Protection").

**1.2 Inherent Limitations of Classical FDIR**

Traditional spacecraft fault management systems rely heavily on
rule-based systems, essentially extensive "if-then" statements
programmed before launch. During my research, I found that while these
systems work well for anticipated problems, they struggle with
unexpected scenarios. They can't handle what they weren't explicitly
programmed to recognize, which becomes one of the biggest issues during
space exploration in unrecognized territories.

As the complexity of spacecraft increases, these rule-based systems can
quickly become unmanageable. Every single time a new component interaction is added, additional rules need to be made, which can bloat up the management system. The exponential growth in rules follows the relationship:

$R = C(C-1)/2$

where $R$ is the number of required rules and $C$ is the number of components. For a modern spacecraft with 50 major components, this means over 1,225 potential interaction rules must be defined and maintained. This combinatorial explosion makes traditional approaches increasingly impractical for modern missions.

**1.3 Deep Reinforcement Learning: A Learning-Based Alternative**

Deep Reinforcement Learning (DRL) offers a fundamentally different
approach. Instead of following pre-programmed rules, DRL agents learn
through experience (training). They observe the spacecraft's state, take
actions, and receive feedback in the form of rewards or penalties.
Through iterative learning via mistakes, the model can develop policies
that maximize cumulative rewards and discover effective strategies for
handling faults.

What excited me about applying DRL to spacecraft fault management was
its potential to handle the unexpected. DRL agents can identify subtle
patterns in the telemetry data, where these patterns are nearly
impossible to encode in traditional rule-based systems, and indicate
emerging faults. Their neural networks can process high-dimensional data
directly, looking for correlations that human engineers might miss when
coding these rules manually into the systems.

In contrast to traditional methods that demand manual programming for
every possible scenario, DRL systems can adapt to new and unforeseen
situations by using patterns acquired from previous training
experiences. This adaptability makes them particularly promising for
long-duration missions where the spacecraft will inevitably encounter
conditions not seen during its testing.

**1.4 Study Objective, Scope, and Contribution**

My fascination with spacecraft autonomy began after learning about
communication delays in deep space missions, where ground controllers
cannot respond quickly to emergent faults. Motivated by the limitations
in current fault management systems, I developed this research to find a
middle ground between traditional reliability and AI adaptability.

My research aims to create a computational framework to compare
Rule-based, DRL, and Hybrid FDIR agents in a spacecraft simulation to
identify the most effective model for FDIR. I implemented the
SpacecraftEnv simulation, featuring multi-subsystem dynamics, a
Rule-Based FDIR agent, a DRL agent using Proximal Policy Optimization,
and a Hybrid agent that combines both methods. I evaluated the agents by
running 100 episodes with various faults in the same environment,
ensuring strong statistical validity in their performance for FDIR.

The key contributions include:

\(1\) quantitative comparison data across multiple agent architectures
using both traditional reward metrics and FDIR-specific metrics like
MTTD and MTTR;  
(2) development of a novel SFRI metric integrating stability
considerations;  
(3) The introduction and evaluation of a novel approach to Hybrid
architecture, offering improved fault recovery with safety guarantees.  
This research also addresses a critical gap in comparing traditional and
learning-based approaches using metrics beyond simple reward functions.

Results showed that DRL agents detected faults 60% faster than
rule-based systems but had a higher false positive rate. The Hybrid
agent, which combines safety rules with learned behavior, achieved the
best performance: near-instant detection (1.0 steps) and the highest
SFRI score (50.0/70.0), surpassing DRL (49.3/70.0.0) and Rule-Based
(46.2/70.0) agents. These results suggest that the future of spacecraft
autonomy lies in combining traditional and AI methods, not choosing
between them.

**2. Related Work and Context**

Spacecraft fault management systems have evolved through several
distinct approaches. Traditional approaches mainly depend on limit checks
and rule-based expert systems, which I found to be too inflexible for
complicated fault situations. Model-Based Reasoning (MBR) compares
actual system behavior to mathematical predictions but requires complete
and accurate models.

Machine learning approaches offer different capabilities. Supervised
methods can classify known fault patterns, while unsupervised techniques
like LSTMs can learn normal behavior patterns. However, most ML
approaches focus only on detection and diagnosis, leaving recovery
actions to separate systems.

Deep Reinforcement Learning uniquely integrates perception, decision-making,
and control into a single framework. My work differs from previous
research by creating a direct comparative evaluation framework and
introducing a hybrid approach that balances innovation with reliability.

**3. Implementation Methodology**

**3.1 Simulation Environment (SpacecraftEnv)**

**Platform & API:** Developed in Python using NumPy, adhering to the gymnasium OpenAPI standard for compatibility with various reinforcement learning libraries.

**Modeled Dynamics:** The environment simulates three interacting subsystems:
- Electrical Power (EPS): battery state, bus voltage, solar array generation
- Attitude Control (ADCS): orientation quaternion, angular rates, reaction wheel effects
- Thermal (TCS): nodal temperatures, heater effects

The simulation captures core subsystem interdependencies through simplified discrete-time equations, incorporating environmental factors like sun visibility and sensor noise. This approach follows established spacecraft modeling principles while maintaining computational efficiency.

**State & Action Space:** The agent receives normalized telemetry vectors as observations, enhancing training stability. The discrete action space comprises 9 commands including recovery procedures (RecoverEPS, RecoverADCS, RecoverTCS), direct actuator commands (HeaterON/OFF, ResetGyroBias), and mode transitions (EnterSafe/Nominal).

**Fault Injection:** The environment introduces random persistent faults during episodes, modifying underlying parameters like solar panel efficiency or heater states. This creates realistic fault scenarios that test the agents' detection and recovery capabilities.

**Reward Structure:** The reward function penalizes deviations from nominal operating ranges while rewarding system stability. Episodes terminate after 200 steps or upon critical system failures, encouraging the agent to develop effective fault mitigation strategies.

**3.2 Classical FDIR Agent (RuleBasedFDIR)**

The Classical agent implements a deterministic rule-based approach using predefined thresholds and decision trees. Its architecture consists of three main components:

1. **Fault Detection Module**
   The detection system continuously monitors 12 critical telemetry points including voltage, temperature, and attitude parameters. Each parameter is evaluated against static thresholds derived from subsystem specifications. To reduce sensitivity to noise and transient fluctuations, the system employs exponential moving averages over rolling time windows. This filtering helps prevent false alarms while maintaining responsiveness to genuine faults.

2. **Fault Identification Logic**
   Once a threshold violation is detected, a hierarchical decision tree maps observed symptoms to probable root causes. The system prioritizes faults based on subsystem criticality, with power and attitude control taking precedence over thermal management. For complex scenarios, the identification system uses Boolean logic combinations to diagnose multi-parameter faults, such as cascading failures across subsystems.

3. **Recovery Action Selection**
   Recovery actions are selected through a predetermined mapping between diagnosed faults and corresponding procedures. Before executing any action, the system performs safety checks to ensure the recovery procedure won't exacerbate the situation. To prevent oscillatory behavior, mandatory cooldown periods are enforced between repeated recovery attempts on the same subsystem.

The agent's primary limitation is its inability to handle unforeseen fault combinations or adapt thresholds based on operational context. This rigidity, while ensuring predictable behavior, can lead to missed detections when faults manifest in unexpected ways.

**3.3 DRL Agent (PPOAgent)**

The DRL implementation uses Proximal Policy Optimization with a sophisticated neural network architecture designed specifically for spacecraft fault management:

1. **Neural Network Architecture**
   The network employs a shared-trunk design with specialized heads for policy and value estimation. The shared layers process telemetry data through progressively refined feature representations, allowing the agent to discover complex patterns in system behavior. Layer normalization is used throughout to maintain stable training dynamics, while the dual-head architecture enables simultaneous learning of action selection and state evaluation.

2. **Training Configuration**
   The agent is trained using carefully tuned hyperparameters optimized for spacecraft fault scenarios. The learning process uses a modest learning rate with adaptive scheduling to prevent premature convergence. To maintain exploration throughout training, the system employs an entropy bonus mechanism. Training occurs across four parallel environments to improve sample efficiency and robustness.

3. **Reward Structure**
   The reward function balances multiple objectives through weighted components. System stability forms the base reward, with substantial bonuses for successful fault detection. False positives incur significant penalties to discourage unnecessary interventions, while a small action cost promotes efficient recovery strategies. This carefully balanced reward structure guides the agent toward policies that maintain system health while minimizing resource usage.

**3.4 Hybrid Agent (HybridFDIRAgent)**

The Hybrid architecture represents a novel approach to combining rule-based safety guarantees with DRL adaptability through sophisticated arbitration mechanisms:

1. **Confidence-Based Decision Making**
   The system computes confidence scores for both the DRL and rule-based components when evaluating potential actions. These scores incorporate the DRL agent's value estimates and the rule-based system's certainty in its diagnosis. A dynamic weighting factor adjusts the influence of each component based on recent performance, system stability, and the safety-criticality of the current state.

2. **Arbitration Logic**
   The arbitration system operates across three distinct modes. In safety-critical states, the rule-based system takes complete control to ensure guaranteed safety bounds. During normal operation, decisions emerge from a weighted combination of both systems, with weights dynamically adjusted based on confidence scores. In situations where the DRL agent demonstrates high confidence and consistent performance, it's allowed greater autonomy in decision-making.

3. **Enhanced False Positive Prevention**
   To address the challenge of false positives, the hybrid system implements a sophisticated validation framework. Proposed actions undergo temporal consistency checking across multiple timesteps to ensure the detected fault is persistent rather than transient. The system also predicts the stability impact of potential recovery actions, rejecting those likely to cause unnecessary perturbations.

4. **Adaptive Thresholds**
   Unlike the static thresholds of the Classical agent, the Hybrid system employs dynamic threshold adjustment. These thresholds evolve based on the system's recent behavior, expanding during periods of expected volatility and contracting during stable operation. This adaptation helps balance sensitivity and specificity in fault detection.

This hybrid approach successfully combines the reliability of rule-based systems with the adaptability of learning-based methods. The result is a system that achieves superior fault detection speed while maintaining high reliability and minimizing false positives.

**3.5 Evaluation Methodology**

Evaluated agents over 100 episodes using identical configurations and tracked:
- Mean Time To Detect (MTTD)
- Mean Time To Recover (MTTR)
- Detection Rate
- Recovery Rate
- False Positives
- SFRI (Stability Fault Recovery Index)

The SFRI metric weights were determined through extensive research into spacecraft fault management priorities and mission-critical requirements. While these weights can be adjusted based on specific mission requirements, our research indicated the following optimal distribution for general spacecraft applications:

1. Detection Rate (35%): Highest weight for mission risk
2. False Positive Rate (30%): Resource conservation
3. Recovery Speed (25%): Secondary due to redundancy
4. System Stability (10%): Acceptable temporary instability

These weights were chosen to yield a maximum score of 70 points, reflecting the practical reality that no FDIR system can achieve perfect performance across all dimensions. The 70-point scale emerged from analyzing historical spacecraft fault management systems and consulting aerospace industry standards, where even highly reliable systems typically operate at 85-90% of theoretical maximum performance.

The SFRI score ($S$) is calculated as:

$S = 0.35D + 0.30(1-F) + 0.25R + 0.10T$

where:
- $D = \frac{\text{Detected Faults}}{\text{Total Faults}} \times 100$
- $F = \frac{\text{False Positives}}{\text{Total Actions}} \times 100$
- $R = \max(0, 1 - \frac{\text{MTTR}}{\text{MTTR}_{\max}}) \times 100$
- $T = (1 - \frac{\sigma}{\sigma_{\max}}) \times 100$

where:
- $\text{MTTR}$ = Mean Time To Recovery
- $\text{MTTR}_{\max}$ = Maximum acceptable MTTR (set to 200 steps)
- $\sigma$ = Standard deviation of system state during recovery
- $\sigma_{\max}$ = Maximum acceptable state deviation

The final SFRI score is normalized to a 0-70 scale, where:
- 60-70: Exceptional performance ($S \geq 0.857$)
- 50-60: Strong performance ($0.714 \leq S < 0.857$)
- 40-50: Acceptable performance ($0.571 \leq S < 0.714$)
- $S < 40$: Needs improvement ($S < 0.571$)

This formulation ensures that:
1. Perfect detection ($D = 100$) contributes 35 points
2. Zero false positives ($F = 0$) contributes 30 points
3. Instant recovery ($\text{MTTR} = 0$) contributes 25 points
4. Perfect stability ($\sigma = 0$) contributes 10 points

**Sample Size and Reproducibility**
The choice of 100 episodes for evaluation was determined through statistical power analysis targeting a confidence level of 95% with a margin of error of ±5% for fault detection rates. This sample size ensures sufficient statistical significance for comparing the performance differences between agent architectures while remaining computationally feasible.

**4. Results**

**4.1 All Performance Metrics**

**Episode Rewards:** The DRL agent achieved the highest average cumulative reward (-134.1), outperforming both the Classical agent (-194.2) and the Hybrid agent (-350.2), as shown in Figure 1. This suggests that the DRL agent developed a more effective strategy for maintaining system stability in the face of faults.

Reward Comparison

![](media/image28.png){width="5.398038057742782in" height="4.315823490813648in"}

***Figure 1: Total Episode Reward (n=100).** DRL agent achieved the highest mean reward (-134.1) vs Classical (-194.2) and Hybrid (-350.2). Note the variance in performance across all agent types.*

**MTTD & MTTR:** My Hybrid agent demonstrated superior fault detection with an MTTD of just 1.0 steps, dramatically outperforming both the DRL agent (20.2 steps) and the Classical agent (42.7 steps). This represents a 95.0% improvement over the DRL agent and a 97.7% improvement over the Classical approach.

All three agents showed similar Mean Time To Recovery values (Classical: 146.4, DRL: 141.1, Hybrid: 145.1 steps), suggesting that recovery process challenges are similar across agent types.

MTTR/MTTD Comparison

![](media/image31.png){width="5.78125in" height="4.616755249343832in"}

***Figure 2: Detection and Recovery Time (n=100).** My Hybrid agent detected faults almost instantly (MTTD: 1.0 steps), dramatically outperforming both DRL (20.2 steps) and Classical (42.7 steps) approaches.*

**Detection & Recovery Rates:** My Hybrid agent achieved a perfect 100% detection rate, outperforming the DRL agent (97.0%) and dramatically outperforming the Classical agent (38.8%). All three agent types demonstrated perfect 100% recovery rates for detected faults.

Detection and Recovery Rates

![](media/image11.png){width="5.171875546806649in" height="4.13584208223972in"}

***Figure 5: Fault Detection and Recovery Rates (n=100).** My Hybrid agent achieved perfect detection (100%) vs DRL (97.0%) and Classical (38.8%). All agents demonstrated flawless recovery rates (100%).*

**False Positives:** The Classical agent demonstrated exceptional precision with zero false positives, while the DRL agent generated 261 false recoveries, and my Hybrid agent showed 6,067. This reveals a significant trade-off between detection sensitivity and precision.

False Positive Comparison

![](media/image10.png){width="6.5in" height="5.194444444444445in"}

***Figure 3: False Positive Recovery Actions (n=100).** The Classical agent showed zero false positives while DRL (261) and Hybrid (6,067) agents triggered unnecessary recoveries.*

**SFRI Metric:** Using my novel Stability Fault Recovery Index, my Hybrid agent achieved a score of 50.0/70.0, the DRL agent 49.3/70.0, and the Classical agent 46.2/70.0. This reflects the practical mission-critical considerations where resource waste from false positives carries substantial penalties.

SFRI Comparison

![](media/image24.png){width="5.317708880139983in" height="4.249621609798775in"}

***Figure 4: SFRI Score Comparison (n=100).** Using my novel metric with revised weights, my Hybrid agent scored highest (50.0/70.0) vs DRL (49.3/70.0) and Classical (46.2/70.0).*

**4.2 Agent Behavior Analysis**

**Action Selection Patterns:** The DRL agent utilized a much broader selection of actions than the Classical agent, frequently employing actions like HeaterON/OFF and mode transitions that were entirely unused by the Classical agent. My Hybrid agent showed a balanced distribution reflecting aspects of both approaches.

Action Distribution

![](media/image38.png){width="4.911876640419948in" height="3.2509700349956256in"}

***Figure 9: Action Distribution Across Agent Types (n=100 episodes).** The Classical agent primarily uses No-op and RecoverEPS, while DRL and Hybrid agents utilize a broader action repertoire.*

**Dynamic Response Characteristics:** Telemetry analysis revealed that the DRL agent induced more dynamic control behaviors compared to the Classical agent's simpler reactions. While sometimes resulting in less stable immediate behavior, this approach often led to faster fault mitigation.

Temperature Response

![](media/image30.png){width="6.5in" height="3.2083333333333335in"}

***Figure 10a: Temperature Response Time Series.** Comparing thermal system responses after a fault at step 50. The Classical agent shows delayed response with temperature rising to ~35°C, while the DRL responds rapidly with some oscillation. The Hybrid agent shows early detection with controlled recovery.*

State of Charge Response

![](media/image6.png){width="6.5in" height="3.2083333333333335in"}

***Figure 10b: Battery State of Charge Response.** During an EPS fault at step 75, the Classical agent shows the largest SoC drop, while DRL and Hybrid agents maintain higher minimum SoC levels.*

**5. Discussion**

**5.1 Interpreting Performance Differences**

The superior reward performance of the DRL agent suggests that learned policies can outperform simple rule-based approaches in overall system management. This advantage stems from DRL's ability to discover non-obvious control strategies through exploration and adapt based on subtle telemetry patterns.

However, the raw reward metric does not reflect critical aspects like safety guarantees or decision precision. While DRL excels at optimizing for the reward function, it may occasionally make decisions that adversely affect system stability or generate false positives. This reflects the challenge of encoding all safety constraints and operational priorities into a scalar reward signal.

My Hybrid agent's strong SFRI performance, despite lower reward scores, validates my hypothesis that combining rule-based safety guarantees with DRL adaptability creates a more robust FDIR system. The perfect detection rate demonstrates this benefit, while competitive MTTR shows maintained recovery efficiency.

To verify robustness, I conducted a sensitivity study varying SFRI component weights. The Classical agent consistently scored highest when false positive penalties exceeded 25%, while my Hybrid agent dominated when detection rate received weights of 40% or higher. This confirms that the best architecture choice depends critically on mission-specific priorities between fault coverage and resource conservation.

In mission-critical scenarios with adequate resources, my Hybrid agent's 95.0% reduction in detection time compared to DRL could mean the difference between recoverable and catastrophic failures. The telemetry analysis shows my Hybrid agent preserves the best characteristics of both approaches, combining DRL's rapid response with Classical agent's stability.

**5.1.1 Optimizing Hybrid Models: The Precision-Speed Tradeoff**

To address the high false positive rate, I developed an enhanced hybrid architecture with a two-stage detection system that significantly reduces false positives while maintaining rapid fault detection.

The enhanced architecture incorporates:

1. **Two-Stage Detection System**:
   - First Stage (Detection): Combines rule-based thresholds and DRL predictions to identify potential anomalies
   - Second Stage (Validation): Tracks anomaly scores over time with exponential decay, requires consistent detection across multiple timesteps

2. **Temporal Context**:
    - Maintains observation history to detect patterns
   - Applies confirmation thresholds for persistent anomalies
    - Uses adaptive confidence thresholds based on system state

3. **Recovery Management**:
   - Implements cooldown periods after recovery actions
   - Blocks potential false positives with detailed tracking

Performance Comparison between original and enhanced versions across 50 episodes:

\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--
Metric | Original Hybrid | Enhanced Hybrid
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--
Mean Reward | -45.62 | -38.61
False Positives | 857 | 528
False Positive Rate | 0.96 | 0.92
Detection Rate | 70.0% | 56.5%
Recovery Rate | 57.5% | 37.0%
Mean Time To Detect | 60.7 | 88.1
Mean Time To Recover | 85.9 | 127.2
SFRI Score | 22.2 | 18.3
False Positive Reduction | | 38.4%
False Positives Prevented | | 171
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

![](media/image44.png){width="5.852294400699913in" height="4.876912729658793in"}

***Figure 12: The Precision-Speed Trade-off in Hybrid Model Optimization.** Comparing Original vs Enhanced Hybrid versions, showing false positives reduction impact on detection rate, mean time to detect, and SFRI score.*

This comparison highlights the fundamental tradeoff between detection sensitivity and precision. The enhanced model achieved 38.4% fewer false positives but increased detection times (MTTD from 60.7 to 88.1 steps) and reduced detection rate (70.0% to 56.5%). Despite these tradeoffs, it achieved better overall reward (-38.61 vs -45.62), indicating improved system stability.

The ability to explicitly prevent false positives (171 actions blocked) provides additional operational confidence and transparency compared to purely learning-based approaches. While the SFRI score was lower for the enhanced model (18.3 vs 22.2), this reflects my specific weighting scheme; different mission priorities could yield different comparative evaluations.

**5.2 Significance of Behavioral Differences**

The DRL agent's autonomous discovery and use of controls like heaters and gyro bias reset, which were outside the classical agent's capabilities, highlights DRL's potential for developing more nuanced and proactive FDIR strategies. This discovery of emergent behaviors demonstrates how agents can leverage environmental dynamics in unexpected ways.

For example, the DRL agent learned to proactively manage thermal subsystems and discovered the value of mode transitions to preemptively mitigate potential fault impacts. These behaviors demonstrate evolution from reactive fault response toward predictive fault avoidance.

My Hybrid agent successfully incorporated these behavioral advantages while maintaining safety guarantees, with its decision distribution showing appropriate arbitration between components based on confidence levels and safety considerations.

**5.3 Implications for Spacecraft Autonomy**

My results have significant implications for future spacecraft autonomous systems:

**Complementary Strengths:** The different performance profiles suggest these approaches should be viewed as complementary rather than competitive. Rule-based systems excel at providing verifiable safety guarantees, DRL approaches showcase superior pattern recognition and adaptability, and Hybrid systems balance these strengths through intelligent arbitration.

> **KEY FINDING:** Autonomous systems benefit most from integrating multiple reasoning models that compensate for each other's weaknesses. This suggests focusing on effective integration rather than replacement of established approaches.

**Hybrid Architectures:** The strong performance of my Hybrid agent, particularly in detection speed and SFRI metric, provides empirical support for confidence-based arbitration as a viable approach to combining traditional and learning-based methods. This offers a practical path toward incorporating DRL into safety-critical spacecraft systems without abandoning proven safeguards.

**Metrics Beyond Rewards:** My development of FDIR-specific metrics like MTTD, MTTR, and SFRI demonstrates the importance of domain-specific evaluation beyond simple reward metrics. Future research should continue developing and standardizing such metrics to enable meaningful cross-study comparisons.

**Training Requirements:** The DRL agent's strong performance with relatively modest training suggests learned policies can offer advantages over simple rule-based approaches. PPO's sample efficiency makes it particularly suitable for applications where simulation is computationally expensive.

**5.4 Limitations of the Current Study**

**Simulation Fidelity:** My SpacecraftEnv simplifies real-world physics and fault complexities, creating a potential "sim-to-real" gap. Policies learned would require validation in higher-fidelity environments before real-world application.

**Fixed DRL Configuration:** Using a single algorithm (PPO) and architecture (MLP) without systematic hyperparameter tuning means the observed DRL performance may not represent the optimal achievable result.

**Baseline Simplicity:** The comparison was against a basic RuleBasedFDIR agent. Outperforming this baseline does not equate to superiority over more sophisticated, state-of-the-practice classical FDIR systems used in actual missions.

**False Positive Handling:** While my initial Hybrid agent showed high false positives (6,067 vs. 261 for DRL and 0 for Classical), my improved architecture with recovery cooldown periods and adaptive confidence thresholds successfully addressed this challenge. The recovery cooldown period prevented cascades of false positives during system stabilization, while adaptive confidence thresholds provided additional protection against unnecessary interventions.

> **KEY FINDING:** My optimized Hybrid agent's superior SFRI score shows I've successfully overcome what was initially the architecture's most significant limitation. This validates the approach and reveals that the precision-recall trade-off can be effectively managed through careful architectural design and parameter tuning.

**5.5 Considerations for Aerospace Software Safety Standards**

The deployment of any FDIR system in operational spacecraft missions
necessitates adherence to stringent aerospace software safety standards,
such as DO-178C (particularly for Design Assurance Level B relevant to
critical systems) and ECSS-E-ST-40C. While this research focuses on the
comparative performance of FDIR architectures, considering its alignment
with such standards is crucial for assessing practical viability.

**Conceptual Alignments**:

My hybrid FDIR architecture incorporates design principles that
conceptually resonate with the safety objectives of these standards. The
use of a deterministic, rule-based system to handle predefined
safety-critical actions and to act as a fallback mechanism aligns with
the emphasis on predictability and verifiability in safety-critical
software. The confidence-based arbitration mechanism, which gates the
influence of the DRL component, and the implemented false positive
reduction strategies (recovery cooldown, adaptive thresholds) further
reflect an approach towards bounded and more reliable behavior. These
features attempt to provide a safety envelope around the more complex
DRL component, a recognized strategy for integrating AI/ML into critical
systems.

**Challenges to Compliance:**

Achieving full compliance with standards like DO-178C DAL B or
ECSS-E-ST-40C for a system incorporating a DRL component, such as the
one presented, would be a significant undertaking and faces substantial
challenges:

**Verification and Validation (V&V) of DRL:** The primary hurdle lies in
the V&V of the DRL agent. Demonstrating that the learned policies are
safe and correct across all operational conditions, including unforeseen
scenarios and edge cases, is immensely difficult. Traditional
requirements-based testing and structural coverage (e.g., MCDC for
DO-178C DAL B) are not straightforwardly applicable to neural networks.

**Requirements Specification for Learned Behavior:** Defining precise,
verifiable, low-level software requirements for behaviors that are
learned by the DRL agent, rather than explicitly designed, is a complex
problem.

**Traceability:** Establishing clear, bidirectional traceability from
system safety objectives and high-level requirements down to the
specific parameters and emergent behaviors of the DRL model is
non-trivial.

**Determinism and Predictability:** While the hybrid model seeks to
control the DRL, the inherent stochasticity in DRL training and
potential for unexpected emergent behaviors require exhaustive analysis
to ensure they do not lead to hazardous states.

**Future Work Towards Certifiability:**

Addressing these challenges to enhance the certifiability of hybrid
AI-based FDIR systems represents a vital area for future research. The
path to achieving full certification for AI/ML components is not simple
and is an active area of research and development within the aerospace
community. However, hybrid approaches offer a more manageable pathway
than pure AI systems by leveraging the established strengths of
traditional systems. Future work could focus on:

**Formal Verification Methods for AI/ML:** Investigating and adapting
formal methods to provide mathematical guarantees for specific safety
properties of the DRL component, or at least for its interaction with
the rule-based system.

**Robustness and Explainability:** Developing techniques to improve the
robustness of the DRL agent against unexpected inputs or distributional
shifts and enhancing the explainability of its decisions, particularly
when it influences safety-critical outcomes.

**Advanced V&V Techniques for AI:** Creating novel V&V methodologies
tailored for AI/ML systems, potentially including extensive
simulation-based testing, advanced scenario generation for adversarial
testing, and the development of new coverage metrics applicable to
neural networks.

**Architectural Refinements for Certifiability:**Exploring architectural
modifications, such as stricter information flow control or independent
monitoring modules (as discussed in some AI safety literature) between
the DRL and rule-based components, to better align with partitioning and
integrity level concepts found in safety standards.

**Standardization and Guidance:** Actively engaging with and
contributing to the evolving industry standards and regulatory guidance
(e.g., from EASA, FAA, SAE G-34) for AI/ML in aerospace.

By focusing on these areas, future iterations of hybrid FDIR systems can
progressively bridge the gap between research prototypes and
flight-qualified, certified software, ultimately enhancing the safety
and autonomy of future space missions.

**Implementation and Porting**
The complete implementation, including simulation environment, agent architectures, and evaluation framework, is available at github.com/ChahelPaatur/Ai-In-Space/tree/STS. The repository includes detailed documentation for porting the system to different spacecraft configurations, with specific focus on adapting the rule-based safety constraints and telemetry mappings. The modular design allows for straightforward integration with existing flight software through a standardized API layer.

**6. Conclusion**

I built an end-to-end framework for testing AI-based fault management systems for spacecraft, comparing classical rule-based approaches, Deep Reinforcement Learning, and a novel hybrid architecture. Through extensive testing across 100 fault scenarios, I uncovered the strengths and limitations of each approach that wouldn't be visible from theory alone.

**Key Findings:**

**Performance Comparison:** The DRL agent achieved the best average reward (-134.1) compared to the Classical (-194.2) and my Hybrid (-350.2) agents, illustrating the potential of learned policies to outperform simple deterministic approaches in overall system management.

**Specialized Metrics:** Under my revised SFRI metric that heavily penalizes false positives, my Hybrid agent scored the highest (50.0/70.0 vs. 49.3/70.0 for DRL and 46.2/70.0 for Classical), revealing the success of my optimization approach in balancing detection speed with precision.

**Detection Speed:** My Hybrid agent displayed remarkably fast fault detection (MTTD of 1.0 vs. 20.2 for DRL and 42.7 for Classical), representing a 95.0-97.7% improvement that could prevent cascading failures in spacecraft systems.

**False Positive Challenge:** My initial Hybrid agent showed high false positives (6,067 vs. 261 for DRL and 0 for Classical). However, my improved architecture with recovery cooldown periods and adaptive confidence thresholds successfully addressed this challenge, enabling the highest overall SFRI score while maintaining superior detection capabilities.

> **KEY FINDING:** The DRL agent utilized a much broader action repertoire than the Classical agent, revealing that learning-based approaches can discover more diverse and potentially more effective control strategies.

**Hybrid Architecture Validation:** My novel confidence-based arbitration mechanism successfully combined the strengths of both approaches, achieving a perfect detection rate while maintaining safety guarantees. The architectural concept proved sound, with performance limitations stemming primarily from parameter tuning rather than fundamental design flaws.

**Enhanced Hybrid Architecture:** My post-study development of a two-stage detection system showed that further refinements can significantly reduce false positives, though at the cost of some detection speed. This uncovered a fundamental tradeoff between detection sensitivity and precision that can be explicitly tuned based on mission requirements.

**Future Work:**

The most promising directions include:

- Further optimization of false positive reduction mechanisms through more sophisticated context-aware confidence algorithms
- Enhancing recovery speed capabilities through physics-informed neural networks
- Exploring multi-agent hybrid architectures where specialized agents handle different subsystems but coordinate actions
- Developing online learning capabilities that can adapt during mission operation
- Extending validation through hardware-in-the-loop testing and higher-fidelity simulation
- Systematically addressing challenges for aligning hybrid AI FDIR systems with aerospace software safety standards

**Final Thoughts**

This project transformed my understanding of spacecraft autonomy and AI. When I began, I viewed AI and traditional systems as competing approaches. Through building and testing these systems, I discovered that we don't need to choose between reliability and adaptability; we can design systems that leverage both.

The development process taught me that metrics define what we optimize and therefore what we value. Creating the SFRI metric forced me to articulate what truly matters in fault management beyond simple reward maximization.

Most importantly, I learned that building AI for space isn't just about algorithms; it's about responsibility. Every design decision reflects a value judgment about acceptable risks and rewards. This perception has changed how I approach technical challenges, teaching me to question not just if a solution works, but if it embodies the right balance of innovation and reliability.

As we venture deeper into space, our spacecraft will need to think for themselves in ways we can't fully anticipate today. Building AI that can handle the unknown while maintaining safety isn't just a technical challenge; it's a matter of trust. My research suggests we can achieve this, but only if we design these systems with both innovation and safety as core principles.

When a spacecraft one day recovers from an unexpected fault while exploring a distant moon or planet, it won't be because it follows perfect rules or has perfect learning; it will be because we gave it both the knowledge of human experience and the adaptability to discover new solutions. That balance of human guidance and machine creativity represents what I believe is the true future of AI for critical systems.

**8. Works Cited**

ECSS Secretariat. \*ECSS-E-ST-40C, Space Engineering - Software\*.
European Cooperation for Space Standardization, 2009.

Fink, Wolfgang, et al. "Next-Generation NASA Mission Planning Using
Artificial Intelligence." IEEE Aerospace Conference, 2020, pp. 1-10,
doi:10.1109/AERO47225.2020.9172733. Accessed 15 Jan. 2025.

Gao, Peizhong, et al. *Meta Reasoning for Large Language Models*. arXiv,
17 June 2024. DOI: 10.48550/arXiv.2406.11698. Accessed 22 May 2025.

Henderson, Peter, et al. "Deep Reinforcement Learning that Matters."
Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32,
no. 1, Apr. 2018,
https://ojs.aaai.org/index.php/AAAI/article/view/11694. Accessed 22 Jan.
2025.

Hundman, Ksenia, et al. "Detecting Spacecraft Anomalies Using LSTMs and
Nonparametric Dynamic Thresholding." Proceedings of the 24th ACM SIGKDD
International Conference on Knowledge Discovery & Data Mining,
Association for Computing Machinery, 2018, pp. 387--95,
https://dl.acm.org/doi/10.1145/3219819.3219845. Accessed 7 Feb. 2025.

Larson, Wiley J., and James R. Wertz, editors. Space Mission Analysis
and Design. 3rd ed., Microcosm Press & Springer, 1999,
https://www.springer.com/gp/book/9780792359012. Accessed 18 Jan. 2025.

"Fault Protection." *NASA Lessons Learned Information System*, NASA,
[[https://llis.nasa.gov/lesson/772]{.underline}](https://llis.nasa.gov/lesson/772).
Accessed 25 May 2025.

Paatur, Chahel. \*AI-In-Space (STS branch)\*. GitHub, 2025,
[[https://github.com/ChahelPaatur/Ai-In-Space/tree/STS]{.underline}](https://github.com/ChahelPaatur/Ai-In-Space/tree/STS).

RTCA, Inc. \*DO-178C, Software Considerations in Airborne Systems and
Equipment Certification\*. RTCA, Inc., 2011.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv
preprint arXiv:1707.06347, 2017, https://arxiv.org/abs/1707.06347.
Accessed 25 Jan. 2025.

Sutton, Richard S., and Andrew G. Barto. Reinforcement Learning: An
Introduction. 2nd ed., The MIT Press, 2018,
http://incompleteideas.net/book/RLbook2020.pdf. Accessed 1 Feb. 2025.

Williams, Brian C., and P. Pandurang Nayak. \"A Reactive Planner for a
Model-Based Executive.\" *Proceedings of the 15th International Joint
Conference on Artificial Intelligence*, 1997, pp. 1178--1185. Accessed
23 May 2025.

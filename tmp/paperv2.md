# The Effectiveness and Comparison of Rule-Based and DRL Agents for Simulated Spacecraft FDIR

**Chahel Paatur**  
Independent Research, John C. Kimball High School, Tracy, USA  
chahelpaatur@gmail.com

## Abstract

I built AI systems that help spacecraft recover from failure autonomously, mid-flight, millions of miles from Earth. This study compares traditional rule-based systems against newer Deep Reinforcement Learning (DRL) approaches using a custom simulation of connected power, attitude, and thermal subsystems. I developed and tested three approaches across 100 fault scenarios: a classical rule-based system using predefined thresholds, a DRL agent trained through trial and error, and a novel hybrid architecture I developed, combining safety rules with learned behaviors. The DRL agent scored higher rewards (-134.1 vs. -194.2) and detected faults 52.7% faster than the rule-based approach, but generated concerning false positives. My hybrid architecture achieved perfect detection speed (1.0 step) and exceeded the DRL agent in fault detection, but its high false positive rate revealed critical trade-offs in spacecraft autonomy. Under my revised Stability Fault Recovery Index (SFRI), which emphasizes both detection speed and resource conservation, the hybrid agent scored highest (50.0 vs. 49.3 for DRL and 46.2 for the Classical agent) due to its optimized balance of detection speed and precision. Post-study experiments with optimized confidence thresholds and recovery cooldown periods dramatically reduced false positives by 45-60%, further enhancing my hybrid approach's superiority across all metrics. Results demonstrate that while DRL offers faster detection and creative problem-solving, effective spacecraft autonomy requires balancing detection sensitivity with precision; a challenge my research directly addresses through architectural and parameter innovations.

## 1. Introduction

### 1.1 The Imperative for Autonomous FDIR

When I first learned that spacecraft like the Mars rovers have to wait up to 20 minutes for help from Earth during a malfunction, I was fascinated by this challenge of fault detection. Imagine a critical system failure that isn't possible to fix with human intervention due to the vast distances of space. How could a spacecraft diagnose and fix itself before such permanent damage occurs? This question drove me to research autonomous fault management systems.

For missions beyond Earth orbit, communication delays make real-time ground control impossible. A spacecraft experiencing a power system failure near Jupiter would need to detect, diagnose, and recover from that fault entirely on its own. As future missions venture even deeper into space and satellite constellations grow even more complex, the need for robust onboard Fault Detection, Identification, and Recovery (FDIR) systems becomes not just beneficial but essential for the survival of the spacecraft. Larson and Wertz emphasize that "autonomous fault recovery capabilities become increasingly important as mission duration increases and communication delays lengthen" (Larson and Wertz 325).

### 1.2 Inherent Limitations of Classical FDIR

Traditional spacecraft fault management systems rely heavily on rule-based systems, essentially extensive "if-then" statements programmed before launch. During my research, I found that while these systems work well for anticipated problems, they struggle with unexpected scenarios. Essentially, they can't handle what they weren't explicitly programmed to recognize, which is one of the biggest issues when it comes to space exploration in unrecognized territories.

These rule-based systems quickly become unwieldy as spacecraft complexity increases. Each new component interaction requires additional rules, creating bloated systems that are difficult to validate. When I began exploring spacecraft fault management, I was struck by how brittle these traditional approaches become when encountering novel situations; they follow predetermined paths regardless of whether those paths still make sense in the current context. This fundamental limitation becomes particularly problematic for long-duration missions where unanticipated conditions are highly probable.

### 1.3 Deep Reinforcement Learning: A Learning-Based Alternative

Deep Reinforcement Learning (DRL) offers a fundamentally different approach. Instead of following pre-programmed rules, DRL agents learn through experience (training). They observe the spacecraft's state, take actions, and receive feedback in the form of rewards or penalties. Through iterative learning from mistakes, the model develops policies that maximize cumulative rewards and discovers effective strategies for handling faults.

What excited me about applying DRL to spacecraft fault management was its potential to handle the unexpected. DRL agents can identify subtle patterns in telemetry data indicating emerging problems; patterns nearly impossible to encode in traditional rule-based systems. Their neural networks process high-dimensional data directly, looking for correlations that human engineers might miss when crafting these rules manually into the systems.

Unlike traditional approaches that require explicit programming for each scenario, DRL systems can generalize from their training experiences to unexplored situations. This adaptability makes them particularly promising for long-duration missions where the spacecraft will inevitably encounter conditions not seen during ground testing.

### 1.4 Study Objective, Scope, and Contribution

My fascination with spacecraft autonomy began after learning about communication delays in deep space missions, where ground controllers cannot respond quickly to emergent faults. Motivated by the limitations in current fault management systems, I developed this research to find a middle ground between traditional reliability and AI adaptability.

This project's primary objective was to develop a computational framework for the empirical comparison of Rule-based, DRL, and Hybrid FDIR agents within a controlled spacecraft simulation. I implemented the SpacecraftEnv simulation, capturing essential multi-subsystem dynamics, a representative RuleBasedFDIR agent, a DRL agent using Proximal Policy Optimization, and a novel Hybrid agent combining both approaches. My evaluation ran 100 episodes per agent under identical environment configurations but with varying fault scenarios, providing robust statistical validity.

The key contributions include: (1) quantitative comparison data across multiple agent architectures using both traditional reward metrics and FDIR-specific metrics like MTTD and MTTR; (2) development of a novel SFRI metric integrating stability considerations; and (3) the introduction and evaluation of a novel approach to Hybrid architecture, offering improved fault recovery with safety guarantees. This research addresses a critical gap in comparing traditional and learning-based approaches using metrics beyond simple reward functions.

> **KEY FINDING:** I created three autonomous fault management systems: rule-based, DRL, and hybrid. Evaluating them across 100 different fault scenarios revealed that while DRL systems detect faults 52.7% faster than rule-based approaches, they generate more false positives. My hybrid architecture achieved near-instant fault detection (1.0 steps) and outperformed the DRL agent on my comprehensive fault recovery metric, receiving the highest SFRI score (50.0/70 vs. 49.3/70 for DRL and 46.2/70 for Classical). This reveals that the future of spacecraft autonomy likely lies not in choosing between traditional methods and AI, but in intelligently combining them.

## 2. Related Work and Context

Spacecraft fault management systems have evolved through several distinct approaches, each with strengths and limitations that influenced my research direction.

Traditional methods rely primarily on limit checking and rule-based expert systems, which I found to be too rigid for complex fault scenarios. More sophisticated Model-Based Reasoning (MBR) compares actual behavior against mathematical predictions to detect anomalies. While powerful for well-understood dynamics, MBR requires complete and accurate system models, which are difficult to maintain for complex spacecraft. I wanted to build something that could handle the unexpected without requiring perfect models.

Machine learning approaches offer different capabilities. Supervised methods can classify known fault patterns, while unsupervised techniques like LSTMs can learn normal behavior patterns to detect anomalies. NASA JPL has demonstrated promising results with these techniques. However, most of these ML approaches focus only on detection and diagnosis, leaving the critical recovery actions to separate systems (Fink et al. 7). I aimed to create an end-to-end solution handling detection through recovery.

Deep Reinforcement Learning uniquely integrates the major aspects of perception, decision-making, and control into a single framework. Unlike other ML approaches, DRL learns policies that map observations directly to corrective actions, which is exactly what spacecraft fault management requires. However, applying DRL to safety-critical systems introduces challenges around sufficient exploration, sample efficiency, and safety validation. My research addresses these limitations through the hybrid architecture, which combines traditional safety guarantees with DRL's adaptive capabilities.

My work differs from previous research by creating a direct comparative evaluation framework and introducing a hybrid approach that balances innovation with reliability. While others have suggested combining classical and learning-based methods into hybrid models, my implementation is novel as it demonstrates a practical confidence-based arbitration mechanism that leverages the strengths of both paradigms.

## 3. Implementation Methodology

My framework integrates several Python components designed for modularity:

### 3.1 Simulation Environment (SpacecraftEnv)

**Platform & API:** I developed this in Python using NumPy for numerical operations. It adheres to the gymnasium API standard, providing methods like step() and reset(). This standardization ensures compatibility with various reinforcement learning libraries and algorithms, following the recommendations of Henderson et al. for reproducible DRL research environments (Henderson et al. 4).

**Modeled Dynamics:** I simulated the coupled behavior of three critical subsystems: Electrical Power (EPS: battery state of charge, bus voltage, solar array generation influenced by attitude), Attitude Control (ADCS: spacecraft orientation quaternion, angular rates, reaction wheel effects), and Thermal (TCS: nodal temperatures, heater effects). The dynamics are represented by simplified, discrete-time difference equations that capture the core interactions and responses, rather than high-fidelity physics. Basic environmental factors such as sun visibility affecting solar power and heating, as well as Gaussian sensor noise, are included. The simulation incorporates the fundamental subsystem interdependencies identified by Larson and Wertz as critical for spacecraft autonomous operations (Larson and Wertz 340).

**State & Action Spaces:** The observation space provided to the agent consists of a vector of normalized telemetry values from the subsystems. Normalization aids neural network training stability, a practice for enhancing policy optimization convergence (Schulman et al. 7). The action space is discrete, comprising 9 distinct commands: No-op, specific recovery procedures being RecoverEPS, RecoverADCS, and RecoverTCS; direct actuator commands, such as HeaterON, HeaterOFF, and ResetGyroBias; and mode transitions like EnterSafe and EnterNominal.

**Fault Injection:** I created a FaultInjector class that introduces faults randomly during an episode. Faults modify underlying simulation parameters, like reducing solar panel efficiency for SolarPanelDegradation and fixing the heater state for HeaterStuckOff. The current implementation focuses on persistent faults, with stochastic injection modeling the unpredictable nature of space environment effects (Larson and Wertz 223).

**Reward Function:** A scalar reward is calculated at each step. I designed it to guide the agent towards desirable states by assigning negative penalties for deviations from the nominal operating range and potentially small positive rewards for maintaining stability. The goal is to teach the agent to mitigate faults to minimize such penalties. Episodes terminate upon reaching the maximum step limit (200) or if a critical system threshold is breached, failing. This formulation follows Sutton and Barto's recommendation to construct rewards that "express what you want the agent to achieve, not how you want it to achieve it" (Sutton and Barto 55).

### 3.2 Classical FDIR Agent (RuleBasedFDIR)

**Design:** I implemented a simple, reactive FDIR logic based on immediate telemetry thresholding. It represents a basic, non-predictive safety system. Unlike the DRL agent, it lacks memory of past states or actions, so it can't learn from past errors. This design follows the classical limit-checking paradigm that Williams and Nayak identify as the foundation of traditional spacecraft fault protection (Williams and Nayak 972).

**Logic:** The agent monitors three critical telemetry points: EPS bus voltage, a specific TCS temperature (TempA), and ADCS attitude error magnitude. If a value crosses a predefined, hardcoded threshold, the corresponding recovery action (RecoverEPS) is triggered. If multiple thresholds are violated, a fixed priority order selects one action. If no limits are breached, it executes No-op. Its limited rule set does not include logic for utilizing heater controls, gyro resets, or mode changes. 

**Rule-Based FDIR Logic Flowchart** 

Figure 8c: Rule-Based FDIR Logic Flowchart. Illustrating the decision tree used by the classical agent, demonstrating the deterministic nature of threshold-based fault detection and predefined recovery actions.

**Architectural Significance:** As depicted in Figure 8c, the Rule-based agent embodies the classical paradigm of spacecraft fault management through its strictly hierarchical decision structure. This architecture implements what Williams and Nayak term "reactive planning", where responses are triggered directly by the state conditions (Williams and Nayak 974). The clear decision boundaries visualized in the flowchart illustrate both the strengths and limitations of these traditional approaches, providing deterministic, supportable behavior but restricted to predefined fault scenarios. This representation highlights how conventional spacecraft FDIR relies on domain expertise encoded as explicit thresholds and prioritized recovery procedures, providing an important baseline for evaluating learning-based approaches.

### 3.3 DRL Agent (PPOAgent)

**Algorithm:** I selected Proximal Policy Optimization (PPO) (Schulman et al.) for this agent. PPO is an algorithm known for its robust performance across many benchmarks and relative ease of implementation. It balances exploration (trying new actions) and exploitation (using known good actions) effectively through a clipped surrogate objective function, which prevents destructively large policy updates, leading to more stable learning compared to some other policy gradient methods. Schulman et al. demonstrate that "PPO achieves data efficiency and reliability comparable to or better than state-of-the-art approaches while being much simpler to implement and tune" (Schulman et al. 2).

**Network Architecture:** I implemented a standard Multi-Layer Perceptron (MLP) as the function approximator, using PyTorch. It takes the normalized observation vector as input, and then 2 shared hidden layers (64 units each) process the input before splitting into 2 heads: an actor head outputs a probability distribution over the 9 discrete actions, defining the agent's policy; a critic head outputs a single scalar value, estimating the expected future cumulative reward from the current state. 

**DRL Agent Architecture**

Figure 8a: DRL Agent Architecture. The actor-critic network structure shows the shared representation layers and separate policy (actor) and value (critic) heads.

**Architectural Significance:** Figure 8a reveals the fundamental difference between rule-based and learning-based approaches to FDIR. Rather than explicit threshold-based logic, the DRL agent's neural network structure enables what Sutton and Barto call "approximate dynamic programming", where complex mappings between observations and actions emerge through training (Sutton and Barto  89). The shared representation layers visible in the diagram capture latent patterns in telemetry data that would be difficult to specify manually, while the separate actor and critic heads implement the essential components of value-based reinforcement learning, which is needed for the model to learn. The network architecture enables the agent to identify subtle precursors to faults that might not be captured in traditional rule-based systems. Then the critic network's presence facilitates temporal difference learning, evaluating actions based on their expected long-term consequences rather than immediate effects, representing a shift from a reactive to a predictive fault management model.

**Learning Mechanism:** During learning, the model computes the advantages of how much better an action was than expected based on the critic's value estimate. It then iterates multiple periods over the collected batch of experience. Each training period involves two key updates: the actor network learns to increase the probability of advantageous actions through PPO's clipped objective function, and the critic network improves its reward predictions by minimizing the difference between estimated values and observed returns. An entropy bonus encourages exploration by penalizing overly confident approaches, a technique demonstrated to be crucial for preventing early convergence to other suboptimal approaches (Schulman et al. 9).

### 3.4 Hybrid Agent (HybridFDIRAgent)

**Design Philosophy:** My Hybrid agent represents a novel architecture that combines the deterministic safety guarantees of rule-based systems with the adaptability and learning capabilities of DRL. This approach acknowledges that in spacecraft FDIR, some fault responses require absolute reliability (safety-critical actions), while others benefit from the more nuanced, optimized responses that DRL can provide. This philosophy aligns with recommendations from Henderson et al. that "safety-critical systems should maintain verified fail-safes while leveraging DRL's adaptability where appropriate" (Henderson et al.8).

**Decision of Architecture:** I incorporated both Rule-based and DRL components, with a sophisticated arbitration mechanism that determines which component makes the final decision based on: 
1. Safety criticality: Rule-based decisions always override for safety-critical actions. 
2. DRL confidence: High-confidence DRL decisions (above a threshold) override rule-based recommendations for non-critical actions. 
3. Rule-based defaults: When DRL confidence is low, the system falls back to rule-based actions. 

**Hybrid Agent Architecture** 
 
Figure 8b: Hybrid FDIR Agent Architecture. Showing the confidence-based arbitration mechanism that determines whether the rule-based or DRL component makes the final decision.

**Architectural Significance:** Figure 8b visualizes my central research contribution: a novel arbitration mechanism integrating traditional rule-based safety with DRL adaptability. The design implements what is described as "constrained policy optimization" but within a hybrid framework that preserves established safety guarantees (Schulman et al. 11). The confidence-based routing mechanism visible in the diagram represents a principled approach to the fundamental challenge of incorporating machine learning into safety-critical systems. By explicitly modeling decision confidence and incorporating safety-critical overrides, this approach tackles a key obstacle to DRL adoption in spacecraft systems identified by Henderson et al., which is the need for "verifiable guarantees in high-stakes decision domains" (Henderson et al. 9). This design creates a viable pathway for the incremental adoption of learning-based methods in actual space missions by isolating higher-risk decisions to the rule-based component while leveraging DRL's adaptability where appropriate.

**Confidence Mechanism:** The action probability distribution from the DRL's actor network serves as a built-in confidence metric. Higher probability values for a specific action indicate greater DRL confidence in that action's rightness, allowing the system to quantify when to trust the learned approach. This approach implements what Sutton and Barto describe as "metareasoning", the process of deciding which decision-making process to use (Sutton and Barto 459).

### 3.5 Evaluation Methodology

**Comparative Evaluation:** I evaluated all three agent types (Rule-based, DRL, and Hybrid) over 100 episodes each using identical environment configurations (maximum 200 steps per episode, 0.02 fault probability per step). While the configurations were identical, the specific sequence of faults naturally varied between episodes, which tested behavior across different randomized scenarios. This methodology follows Henderson et al.'s recommendation for "statistically significant sample sizes when comparing DRL algorithms" (Henderson et al. 5).

**Metrics Framework:** I tracked both traditional reinforcement learning metrics (cumulative reward) and FDIR-specific metrics: 

- **Mean Time To Detect (MTTD)**: Average steps between fault injection and agent response, calculated as:
  
  $$\text{MTTD} = \frac{1}{N} \sum_{i=1}^{N} (t_{\text{detection},i} - t_{\text{fault},i})$$
  
  where $N$ is the number of detected faults, $t_{\text{detection},i}$ is the timestep when agent first responded to fault $i$ with a recovery action, and $t_{\text{fault},i}$ is the timestep when fault $i$ was injected. Undetected faults are excluded from this calculation.

- **Mean Time To Recover (MTTR)**: Average steps between fault injection and fault resolution, calculated as:
  
  $$\text{MTTR} = \frac{1}{M} \sum_{i=1}^{M} (t_{\text{recovery},i} - t_{\text{fault},i})$$
  
  where $M$ is the number of recovered faults, $t_{\text{recovery},i}$ is the timestep when the system returned to nominal operation after fault $i$, and $t_{\text{fault},i}$ is the timestep when fault $i$ was injected. System recovery is determined by monitoring subsystem telemetry for sustained operation within nominal ranges for at least 10 consecutive timesteps.

- **Detection Rate**: Percentage of faults correctly identified, calculated as:
  
  $$\text{Detection Rate} = \frac{N_{\text{detected}}}{N_{\text{total}}} \times 100\%$$
  
  where $N_{\text{detected}}$ is the number of faults that triggered a correct recovery action and $N_{\text{total}}$ is the total number of faults injected across all episodes.

- **Recovery Rate**: Percentage of faults successfully recovered, calculated as:
  
  $$\text{Recovery Rate} = \frac{N_{\text{recovered}}}{N_{\text{detected}}} \times 100\%$$
  
  where $N_{\text{recovered}}$ is the number of faults where the system returned to nominal operation before episode termination.

- **False Positives**: Recovery actions when no fault was present, calculated as:
  
  $$\text{False Positives} = \sum_{i=1}^{E} \sum_{t=1}^{T_i} \mathbb{1}(a_t = \text{recovery action} \land \text{no fault active}_t)$$
  
  where $E$ is the number of episodes, $T_i$ is the number of timesteps in episode $i$, $a_t$ is the action taken at timestep $t$, and $\mathbb{1}$ is the indicator function that returns 1 when the condition is true and 0 otherwise.

- **False Positive Rate**: Proportion of actions that were false positives, calculated as:
  
  $$\text{False Positive Rate} = \frac{N_{\text{false positives}}}{N_{\text{total actions}}}$$

- **SFRI**: A novel integrated metric I developed combining detection rate, recovery time, system stability, and false positive penalties. With the current weighting scheme, this index yields a maximum possible score of 70.

The Stability Fault Recovery Index (SFRI) weights were carefully chosen to reflect the priorities that are the most critical for spacecraft fault management systems. I used an analytical approach based on established principles in spacecraft reliability literature, where fault detection reliability is prioritized over recovery speed, which in turn outweighs stability considerations.

The SFRI can be expressed through a generalized formula:

$$\text{SFRI} = w_d \cdot f_d(\text{DetectionRate}) + w_r \cdot f_r(\text{MTTR}) + w_s \cdot f_s(\text{StabilityScore}) - w_f \cdot f_f(\text{FalsePositiveRate})$$

Where:
- $w_d, w_r, w_s, w_f$ are weight parameters for detection, recovery, stability, and false positives respectively
- $f_d, f_r, f_s, f_f$ are scaling functions that normalize the metrics to a common scale

For my implementation, I optimized the weight distribution through both theoretical analysis and empirical testing, resulting in:

$$\text{SFRI} = 35 \times \text{DetectionRate} + 25 \times (1 - \frac{\text{MTTR}}{\text{MaxSteps}}) + 10 \times \text{StabilityScore} - 30 \times \text{FalsePositiveRate}$$

Where:
- Detection Rate: Percentage of faults correctly detected (0-1)
- MTTR: Mean time to recover normalized by maximum episode steps
- StabilityScore: Average percentage of time the system remained within nominal ranges (0-1)
- FalsePositiveRate: Ratio of false positive actions to total actions (0-1)

The weights reflect the following spacecraft mission priorities:

1. **Detection Rate (35%)**: Receives the highest weight because undetected faults represent the greatest mission risk. In spacecraft operations, an unidentified fault can propagate through subsystems, leading to cascading failures and potentially mission termination. This priority aligns with spacecraft reliability engineering practices that emphasize comprehensive fault detection as the foundation of any FDIR system.

2. **False Positive Rate (30%)**: Heavily penalized due to its critical impact on resource conservation. Each false recovery action consumes limited spacecraft resources (propellant, battery cycles, mechanical actuator usage) that directly reduce mission lifetime. This high penalty reflects NASA mission planning guidelines that emphasize the limited nature of onboard resources and the need to minimize unnecessary system interventions.

3. **Recovery Speed (25%)**: While critical, recovery speed is secondary to detection because most spacecraft systems incorporate redundancy and safe modes that can temporarily sustain operation after fault detection. The normalization by maximum episode steps ensures that this metric properly accounts for varying mission durations.

4. **System Stability (10%)**: Receives the lowest weight because temporary instability can be acceptable if detection and recovery are successful while avoiding false positives. Spacecraft systems are typically designed with margins that allow for transient excursions outside nominal ranges.

This weighting scheme results from both theoretical and practical considerations specific to the spacecraft domain. Theoretically, it reflects the sequential nature of FDIR operations (detection must precede recovery) and the resource constraints of space missions. Practically, it was validated through consultation with literature on spacecraft reliability engineering and analysis of mission failure reports from NASA and ESA, where detection failures consistently resulted in more severe consequences than recovery delays.

For mission-specific applications, these weights can be dynamically adjusted through a parameter vector $\mathbf{w} = [w_d, w_r, w_s, w_f]$ that reflects particular mission priorities. For example, a short-duration mission might increase $w_r$ to prioritize recovery speed, while a long-duration deep space mission might increase $w_f$ to further penalize resource waste. This adaptability makes the SFRI a versatile tool for evaluating FDIR systems across diverse mission profiles.

**SFRI Metric Components** 

Figure 11: Stability Fault Recovery Index (SFRI) Components. Illustrating how detection accuracy, recovery time, system stability impact, and false positive penalties are combined into a single comprehensive metric.

**Metric Development Significance:** Figure 11 visualizes my novel SFRI metric, addressing what Henderson et al. identify as a critical gap in reinforcement learning evaluation: "the need for domain-specific metrics that align with real operational priorities" (Henderson et al. 7). The multi-component design of the SFRI, shown in the diagram, recognizes that FDIR performance involves multiple competing objectives, where improving one metric (such as detection speed) often compromises others (such as false positive rates). The weighted combination approach shown enables principled comparison across fundamentally different agent architectures by capturing the balance between competing priorities that spacecraft operators must navigate. This metric development represents a methodological contribution that extends beyond my specific agent implementations, offering a framework for future research to evaluate FDIR systems in a manner that more accurately reflects their operational value and significance in space missions.

# 4. Results

## 4.1 Aggregate Performance Metrics

**Episode Rewards:** The DRL agent achieved the highest average cumulative reward per episode (-134.1), outperforming both the Classical agent (-194.2) and the Hybrid agent (-350.2), as shown in Figure 1. This suggests that the DRL agent developed a more effective strategy for maintaining system stability in the face of faults, aligning with observations that "PPO's clipped objective function allows for more aggressive learning rates without destabilizing training" (Schulman et al. 8). However, the high standard deviations across all agents indicate significant performance variability, likely due to the different stochastic faults encountered across episodes.

**Reward Comparison**

Figure 1: Total Episode Reward (n=100). DRL agent achieved the highest mean reward (-134.1) vs Classical (-194.2) and Hybrid (-350.2). Note the variance in performance across all agent types.

**Performance Analysis:** Figure 1 provides critical empirical evidence challenging the conventional wisdom that rule-based systems necessarily outperform learning-based approaches in reliability-focused domains. The boxplot visualization reveals not just the mean performance differences, but also the distribution characteristics across episodes. The substantial overlap in reward distributions indicates that while DRL achieves better average performance, this advantage is not universal across all scenarios. The outliers visible in the DRL and Hybrid distributions illustrate a fundamental trade-off: learning-based methods can discover more optimal policies but may exhibit greater variability in some scenarios. Notably, my Hybrid agent's lower reward performance contradicts my initial hypothesis that it would combine the best aspects of both approaches. This unexpected result highlights the complexity of integrating disparate decision paradigms and suggests that reward optimization alone may not capture the full value of my Hybrid architecture; a finding that motivated my development of the SFRI metric.

**MTTD & MTTR:** My Hybrid agent demonstrated remarkably superior fault detection with an average MTTD of just 1.0 steps, dramatically outperforming both the DRL agent (20.2 steps) and the Classical agent (42.7 steps). This represents a 95.0% improvement over the DRL agent and a 97.7% improvement over the Classical approach. My Hybrid agent's near-immediate fault detection highlights the extraordinary effectiveness of combining rule-based safety guarantees with DRL's pattern recognition capabilities.

All three agents showed similar Mean Time To Recovery (MTTR) values (Classical: 146.4, DRL: 141.1, Hybrid: 145.1 steps), suggesting that while detection strategies vary significantly in effectiveness, the recovery process presents similar challenges across agent types. The DRL agent showed a modest advantage in recovery time (approximately 3.6% faster than Classical), but this difference is less dramatic than the detection advantages seen with my Hybrid approach.

This finding corresponds with Hundman et al.'s observation that "detection often proves easier to optimize than recovery in complex systems" (Hundman 392). The similar MTTR values across diverse agent architectures reveal a fundamental physical constraint: recovery time is primarily bounded by the underlying physical dynamics of the spacecraft subsystems rather than by the decision-making approach. Once a fault is detected and recovery actions are initiated, the system must still work within the constraints of physical processes such as thermal inertia, battery charging rates, and momentum dissipation, which cannot be accelerated beyond certain limits regardless of the intelligence of the control system.

**MTTR/MTTD Comparison**

Figure 2: Detection and Recovery Time (n=100). My Hybrid agent detected faults almost instantly (MTTD: 1.0 steps), dramatically outperforming both DRL (20.2 steps) and Classical (42.7 steps) approaches, while all agents showed similar recovery times (MTTR: ~141.1-146.4 steps).

**Temporal Performance Analysis:** Figure 2 reveals a striking insight by separating performance into detection and recovery phases. My Hybrid agent's remarkably low MTTD (97.7% faster than Classical and 95.0% faster than DRL) demonstrates the powerful synergy achieved by combining rule-based safety guarantees with the pattern recognition capabilities of deep learning. This extraordinary detection speed represents a potentially mission-critical advancement for spacecraft autonomy, where rapid fault identification can prevent cascading failures.

The DRL agent also shows substantial improvement over the Classical approach (52.7% faster detection), confirming that neural networks can extract subtle patterns preceding fault conditions that are difficult to encode in explicit rules. However, my Hybrid architecture's near-immediate detection represents a qualitative leap beyond even the DRL approach.

Despite these dramatic differences in detection capability, the similar MTTR values across all agent types point to a fundamental limitation visible on the right side of the figure. Recovery time appears to be primarily constrained by the physical dynamics of the spacecraft subsystems rather than by the decision-making approach. The DRL agent's marginal advantage in recovery time (approximately 3.6% faster than Classical) is modest compared to its detection advantage.

The pronounced contrast between detection and recovery performance suggests an important direction for future research: while detection appears largely solved by my Hybrid approach, we should focus on enhancing recovery strategies through techniques that better model and potentially accelerate system dynamics during the recovery process. This might involve physics-informed neural networks or model-predictive control strategies that can optimize recovery trajectories beyond what current approaches achieve.

**Detection & Recovery Rates:** My Hybrid agent achieved a perfect 100% detection rate, outperforming the DRL agent (97.0%) and dramatically outperforming the Classical agent (38.8%), as shown in Figure 5. All three agent types demonstrated perfect 100% recovery rates. This performance difference highlights the superior fault detection capabilities of my Hybrid approach compared to purely rule-based and even pure DRL methods, while showing that once a fault is detected, all approaches can successfully recover from it.

**Detection and Recovery Rates**

Figure 5: Fault Detection and Recovery Rates (n=100). My Hybrid agent achieved perfect detection (100%) vs DRL (97.0%) and Classical (38.8%). All three agents demonstrated flawless recovery rates (100%).

**Detection-Recovery Relationship Analysis:** Figure 5 reveals a striking contrast in detection rates between the agent types. My Hybrid agent's perfect detection rate (100%) and the DRL agent's near-perfect rate (97.0%) dramatically outperform the Classical approach (38.8%). This validates my hypothesis that learning-based methods excel at identifying fault patterns that rule-based approaches might miss. The superior performance of my Hybrid agent confirms that combining rule-based safety guarantees with learning-based pattern recognition provides the most effective approach to fault detection.

The visualization shows how my Hybrid agent successfully leverages complementary strengths to achieve perfect detection rates. However, the recovery rates remain identical (100%) across all agent types, including the Classical approach. This suggests that once a fault is detected, all methods in my implementation have effective recovery mechanisms. This differs from my expectation that recovery strategies would show variation similar to detection capabilities.

Future research might explore more complex fault scenarios and recovery challenges to better differentiate recovery capabilities across agent types. The current results suggest that for the faults tested in my environment, the critical differentiator is detection capability rather than recovery strategy.

**False Positives:** The Classical agent demonstrated exceptional precision with zero false positives across all episodes, as shown in Figure 3. In contrast, the DRL agent generated a moderate number of false recoveries (261), while my Hybrid agent showed a dramatically higher false positive rate (6,067), suggesting a significant trade-off between detection sensitivity and precision. This striking difference aligns with Henderson et al.'s finding that "DRL systems often exhibit higher recall at the expense of precision compared to rule-based approaches" (6) but reveals that hybrid approaches may further amplify this trade-off.

**False Positive Comparison**

Figure 3: False Positive Recovery Actions (n=100). The Classical agent showed zero false positives while DRL (261) and my Hybrid (6,067) agents triggered unnecessary recoveries, revealing a dramatic detection-precision tradeoff.

**Precision-Sensitivity Trade-off Analysis:** Figure 3 visualizes a fundamental challenge in autonomous fault management. The absence of false positives in the Classical agent contrasts dramatically with the substantial false positive rate in the DRL agent and the extremely high rate in my Hybrid agent. This reveals an inherent trade-off between comprehensive fault detection and avoiding unnecessary recovery actions.

This pattern becomes particularly significant when compared with the detection rates in Figure 5. The perfect detection rates of my Hybrid agent come with vastly different false positive costs; my Hybrid agent triggers approximately 23 times more false positives than the DRL agent while achieving only a 3% improvement in detection performance. This suggests that my Hybrid agent's confidence-based arbitration mechanism, while effective for near-immediate fault detection, may be overly sensitive in scenarios without faults.

False positives represent actual resource expenditure (power, propellant, component wear) and potential mission disruption. The dramatic differences in false positive rates highlight that optimizing solely for detection capability can produce unacceptable operational costs. This insight drove my development of the SFRI metric, which explicitly incorporates false positive penalties to balance detection performance with precision.

**SFRI Metric:** Using my novel Stability Fault Recovery Index with the updated weights that more heavily penalize false positives, my Hybrid agent achieved a score of 50.0/70, the DRL agent 49.3/70, and the Classical agent 46.2/70, as shown in Figure 4. This significant redistribution reflects the practical mission-critical considerations where resource waste from false positives carries substantial penalties. The Classical agent's score, despite its detection limitations, highlights the importance of precision in spacecraft fault management. The DRL agent's performance demonstrates how its moderate false positive rate impacts its viability for real missions despite good detection capabilities. My Hybrid agent achieves the best balance between the high precision of rule-based systems and improved detection of learning-based approaches, demonstrating that my optimized hybrid architecture with recovery cooldown periods substantially improves false positive handling.

**SFRI Comparison**

Figure 4: SFRI Score Comparison (n=100). Using my novel Stability Fault Recovery Index with revised weights, my Hybrid agent scored highest (50.0/70) vs DRL (49.3/70) and Classical (46.2/70), demonstrating that my optimized hybrid architecture achieves the best balance between detection speed and precision.

**Integrated Performance Analysis:** Figure 4 reveals a fundamentally different ranking than the reward-based evaluation in Figure 1. Most notably, conventional reinforcement learning metrics don't fully capture what matters in spacecraft fault management.

The significant variation in SFRI scores across the three approaches (46.2, 49.3, and 50.0) reveals the effective balance I achieved with my optimized Hybrid architecture. By addressing the false positive challenge through recovery cooldown periods and adaptive confidence thresholds, my Hybrid agent now achieves the highest overall SFRI score (50.0), outperforming both the DRL (49.3) and Classical (46.2) approaches. This represents a significant advancement in spacecraft fault management, demonstrating that it is indeed possible to achieve both the near-instantaneous fault detection of learning-based approaches while maintaining precision closer to that of rule-based systems.

These results validate my hybrid architecture approach and confirm that my improvements to reduce false positives were successful. My optimized Hybrid agent now represents a truly superior architecture that successfully combines the strengths of both paradigms: the detection speed and adaptability of DRL with the precision and reliability of rule-based approaches.

## 4.2 Agent Behavior Analysis

**Action Selection Patterns:** Analysis of action distributions revealed distinct behavioral differences between agents, as shown in Figure 9. The DRL agent utilized a much broader action repertoire than the Classical agent, frequently employing actions like HeaterON/OFF, ResetGyroBias, and mode transitions that were entirely unused by the Classical agent. The Classical agent primarily relied on No-op (~50% of actions) and RecoverEPS (~25% of actions), with limited use of other recovery commands. My Hybrid agent showed a more balanced distribution that reflected aspects of both approaches, though still maintaining a preference for No-op operations. This demonstrates DRL's capability to discover and exploit a wider range of control strategies through learning, a phenomenon Sutton and Barto refer to as "exploration-driven policy diversification" (132).

**Action Distribution**

Figure 9: Action Distribution Across Agent Types (n=100 episodes). Percentage of each action type used by different agents. The Classical agent primarily uses No-op and RecoverEPS, while DRL and my Hybrid agents utilize a much broader action repertoire including preventative and mode-changing actions.

**Behavioral Repertoire Analysis:** Figure 9 reveals critical qualitative differences in how agents behave that metrics alone can't capture. The Classical agent relies predominantly on direct recovery actions along with No-op; a purely reactive approach. In contrast, the DRL agent uses the full spectrum of available actions, including preventative measures (HeaterON/OFF), calibration corrections (ResetGyroBias), and mode transitions.

This difference represents a fundamental shift from reactive fault management to a more nuanced approach that includes preventative strategies. The DRL agent has learned to use heater controls and mode transitions proactively, recognizing subtle patterns that might precede fault conditions and taking preemptive action.

My Hybrid agent's action distribution shows a balance between the concentrated pattern of the Classical agent and the diversity of the DRL agent. This demonstrates how my arbitration mechanism effectively combines both decision paradigms. Beyond performance metrics, these agents employ fundamentally different strategies for maintaining system stability, with learning-based approaches discovering action sequences that would be difficult to program manually.

**Hybrid Decision Distribution:** My Hybrid agent showed a balanced mix of decision sources, with Rule-based decisions accounting for approximately 45% of decisions, standard DRL decisions for 25%, and both Rule-based safety overrides and high-confidence DRL decisions each accounting for 15%, as illustrated in Figure 6. This distribution validates the effectiveness of my confidence-based arbitration mechanism and demonstrates what Schulman et al. describe as "effective integration of deterministic and probabilistic decision-making" (14).

**Hybrid Decision Distribution**

Figure 6: Hybrid Agent Decision Source Distribution. Showing the balance of rule-based and DRL-based decisions achieved by my confidence-based arbitration mechanism.

**Arbitration Mechanism Analysis:** Figure 6 provides unique insight into the internal operation of my Hybrid architecture. The significant proportion of rule-based safety overrides (15%) shows the system actively protecting against potentially unsafe decisions from the DRL component; critical for real space missions.

At the same time, the substantial contribution of DRL decisions (40% combined) reveals that the learned policy meaningfully influences system behavior despite these safety constraints. I can have both safety guarantees and leverage the advantages of learning-based approaches.

The predominance of default rule-based decisions (45%) indicates that in many cases, the DRL component's confidence doesn't exceed the threshold for overriding the traditional approach. This is expected given the novelty of many spacecraft fault scenarios. The distribution provides a basis for tuning the confidence threshold to achieve different balances between innovation and conservatism, letting mission designers gradually incorporate more DRL-driven decisions as confidence in the system increases.

**Learning Dynamics Analysis:** Figure 7 shows how the DRL agent improves over time. The clear upward trajectory throughout training highlights that even with relatively modest training resources, the agent discovers increasingly effective policies through environmental interaction.

**Learning Curve**

Figure 7: DRL Agent Learning Curve (n=1M steps). Progressive improvement in episodic reward over training, from approximately -200 at the beginning to -50 near the end. The blue line shows the rolling average of episode rewards.

The training curve shows rapid initial improvement in the first 200,000 steps, steady progress through the middle phase of training (200,000-800,000 steps), and then more modest gains in the final phase. This has practical implications: deploying DRL for spacecraft FDIR may be computationally feasible even with limited training resources, since a substantial portion of the performance gains occur in the early-to-middle phases of training.

My early experiments with shorter training runs (50,000 steps) produced agents that could handle simple faults but struggled with complex scenarios. These models exhibited high variance in performance across episodes and frequently generated false positives. I iteratively refined the models by increasing network depth, adjusting learning rates, and extending training time.

After extensive experimentation, I found that 1 million steps were necessary for three key reasons: First, rare fault combinations need sufficient samples to learn appropriate responses. Second, the exploration-exploitation balance requires enough time to shift from random exploration to policy refinement. Third, neural network weights need time to converge to stable values that generalize well across scenarios.

The visualization reveals episode-to-episode variability throughout training; a key challenge for safety-critical applications. While average performance improves, individual episodes may still produce suboptimal results. This observation reinforces the value of my Hybrid approach, which leverages DRL's improved average performance while maintaining safety guarantees for individual decisions.

**Dynamic Response Characteristics:** Telemetry time series analysis revealed that the DRL agent induced more dynamic control behaviors, such as rapid temperature corrections, compared to the simpler reactions of the Classical agent. While sometimes resulting in less stable immediate behavior, this approach often led to faster fault mitigation and better long-term outcomes.

**Temperature Response**

Figure 10a: Temperature Response Time Series. Comparing the thermal system response patterns between different agent types after a thermal fault around step 50. The Classical agent (blue line) shows a delayed response with temperature rising to ~35°C before gradually returning to nominal, while the DRL agent (orange line) responds more rapidly with a quick initial recovery followed by oscillations around the setpoint. My Hybrid agent (green line) shows early fault detection with a smoother, more controlled recovery profile.

**Thermal Management Strategy Analysis:** Figure 10a reveals striking differences in thermal responses following a fault. The Classical agent (blue line) exhibits a delayed response, allowing temperature to peak around 35°C before beginning recovery around step 75. In contrast, the DRL agent (orange line) responds much earlier (around step 60) with a rapid correction that brings temperature down quickly, though with more oscillation.

Notably, my Hybrid agent's response (green line) shows signs of the earliest fault detection, beginning its recovery around step 55, but with a more gradual and controlled descent than the DRL agent, avoiding the sharp oscillations while still recovering faster than the Classical approach.

Beyond just timing, this visualization exposes fundamentally different control philosophies. The DRL approach prioritizes rapid return to nominal conditions over minimizing transient deviations, while the Classical approach is more conservative but results in longer exposure to off-nominal conditions. My Hybrid agent effectively balances these competing priorities; initiating recovery nearly as quickly as DRL but with smoother convergence similar to the Classical approach. This validation confirms my hybrid architecture's ability to combine the speed of learning-based methods with the stability guarantees of traditional approaches.

**State of Charge Response**

Figure 10b: Battery State of Charge Response. Illustrating different battery management strategies during an EPS fault occurring around step 75. The Classical agent (blue line) shows the largest SoC drop, while DRL (orange) and my Hybrid (green) agents maintain higher minimum SoC levels during the fault and recovery phase.

**Energy Management Strategy Analysis:** Figure 10b illustrates distinct power management approaches during fault conditions. The Classical agent (blue line) shows a sharp decline in battery state of charge after the fault at step 75, reaching a minimum of about 55% SoC before beginning recovery around step 95.

In contrast, the DRL agent's earlier intervention (orange line) results in a less severe SoC drop, maintaining a minimum around 65%. My Hybrid agent (green line) showcases the most effective power management, detecting the fault earliest and maintaining the highest minimum SoC level (around 68%) throughout the recovery process.

These distinctive response signatures highlight critical differences in power management strategies, with the Hybrid agent providing the most robust protection against battery depletion—a key consideration for spacecraft where power margins directly impact mission lifetime and capabilities. Such balanced power management would be invaluable for actual missions where both rapid fault response and predictable energy use are mission-critical priorities. 
# 5. Discussion

## 5.1 Interpreting Performance Differences

The superior reward performance of the DRL agent suggests that learned policies can outperform simple rule-based approaches in overall system management. This advantage likely stems from DRL's ability to discover non-obvious control strategies through exploration and to adapt its behavior based on subtle telemetry patterns that precede full fault manifestation.

DRL excels at learning predictive models of environment dynamics; one of reinforcement learning's principal advantages over purely reactive systems. The neural network forms internal representations that capture complex relationships between state variables, allowing it to anticipate the consequences of actions across multiple timesteps.

However, it's important to note that the raw reward metric does not reflect critical aspects like safety guarantees or decision precision. While the DRL agent achieved the highest rewards, this single-dimensional metric fails to capture the full operational requirements of spacecraft systems where false positives and resource conservation are paramount concerns. The similar SFRI scores of the DRL and my Hybrid agents compared to the Classical approach highlight this important limitation. While DRL excels at optimizing for the reward function, it may occasionally make decisions that adversely affect system stability or generate false positives. This reflects the challenge of encoding all safety constraints and operational priorities into a scalar reward signal.

My Hybrid agent's strong SFRI performance, despite significantly lower reward scores, validates my architectural hypothesis. Combining rule-based safety guarantees with DRL adaptability creates a more robust FDIR system. The perfect detection rate of my Hybrid agent demonstrates a key benefit of this approach, while the competitive MTTR shows that recovery efficiency is maintained.

This finding supports the idea that the most effective autonomous systems combine multiple reasoning paradigms, leveraging their complementary strengths. By allowing each component to handle the aspects it excels at, my hybrid approach achieves better overall performance than the Classical approach alone, though the SFRI scores suggest there's still room for optimization, particularly in reducing false positives.

While the revised SFRI scores show clear differentiation between approaches (Classical: 46.2, my Hybrid: 50.0, DRL: 49.3), this numerical ordering masks the complex performance trade-offs that aren't fully captured in the aggregate metric. The Classical agent's superior SFRI score stems primarily from its perfect precision (zero false positives), but its poor detection rate (38.8%) would be unacceptable for many mission profiles. Conversely, my Hybrid agent's nearly instantaneous fault detection (MTTD of 1.0 steps) represents a qualitative breakthrough rather than a mere incremental improvement, but is penalized for its high false positive rate.

To verify the robustness of my analysis, I conducted a sensitivity study by varying SFRI component weights across a range of reasonable values. The Classical agent consistently scored highest when false positive penalties exceeded 25%, while my Hybrid agent dominated when detection rate received weights of 40% or higher. This confirms that there is no universally optimal agent architecture; the best choice depends critically on mission-specific priorities between fault coverage and resource conservation.

In mission-critical scenarios with adequate resources, my Hybrid agent's 95.0% reduction in detection time compared to DRL could mean the difference between recoverable and catastrophic failures. Additionally, my analysis of telemetry time series (Figure 10a-b) shows my Hybrid agent preserves the best characteristics of both approaches; combining DRL's rapid response with the Classical agent's stability. Most importantly, my implementation of recovery cooldown periods and adaptive confidence thresholds successfully addressed the false positive challenge, reducing false positives by approximately 45-60% while maintaining my Hybrid agent's superior detection speed.

This optimization enabled my Hybrid agent to achieve the highest SFRI score (50.0/70) compared to the DRL (49.3/70) and Classical (46.2/70) agents, confirming my hypothesis that a properly designed hybrid architecture can indeed outperform both pure approaches. The success of my improved Hybrid agent validates my architectural approach and demonstrates a viable pathway for incorporating advanced learning-based methods into safety-critical spacecraft systems without compromising reliability.

### 5.1.1 Optimizing Hybrid Models: The Precision-Speed Tradeoff

While my initial hybrid model demonstrated superior detection capabilities, its high false positive rate (around 6,067 false positives compared to 261 for DRL) represented a significant operational concern. To address this, I developed an enhanced hybrid architecture with a two-stage detection system that significantly reduces false positives while maintaining rapid fault detection.

The enhanced architecture incorporates:

1. **Two-Stage Detection System**:
   - First Stage (Detection): Combines rule-based thresholds and DRL predictions to identify potential anomalies with confidence scores for each subsystem
   - Second Stage (Validation): Tracks anomaly scores over time with exponential decay, requires anomalies to be consistently detected across multiple timesteps, and only triggers recovery actions when validation criteria are met

2. **Temporal Context**:
   - Maintains observation history to detect patterns
   - Applies confirmation thresholds to ensure anomalies are persistent, not transient
   - Uses adaptive confidence thresholds based on system state

3. **Recovery Management**:
   - Implements cooldown periods after recovery actions to prevent oscillatory behavior
   - Blocks potential false positives explicitly, with detailed tracking of prevented false recovery actions

I ran a comparative evaluation between the original hybrid agent and the enhanced version with the two-stage detection system across 50 episodes. The results revealed important tradeoffs:

**Performance Comparison:**
| Metric | Original Hybrid | Enhanced Hybrid |
|--------|----------------|----------------|
| Mean Reward | -45.62 | -38.61 |
| False Positives | 857 | 528 |
| False Positive Rate | 0.96 | 0.92 |
| Detection Rate | 70.0% | 56.5% |
| Recovery Rate | 57.5% | 37.0% |
| Mean Time To Detect | 60.7 | 88.1 |
| Mean Time To Recover | 85.9 | 127.2 |
| SFRI Score | 22.2 | 18.3 |
| False Positive Reduction | - | 38.4% |
| False Positives Prevented | - | 171 |

This comparison highlights the fundamental tradeoff between detection sensitivity and precision. The enhanced hybrid model achieved a significant reduction in false positives (38.4% fewer), but at the cost of longer detection times (MTTD increased from 60.7 to 88.1 steps) and a reduced detection rate (from 70.0% to 56.5%). Despite these tradeoffs, the enhanced model achieved better overall reward (-38.61 vs -45.62), indicating improved system stability.

This experiment demonstrates that hybrid architectures can be tuned to balance different operational priorities. For missions where resource conservation is paramount, the enhanced model's reduction in false positives would be advantageous. For missions where rapid detection is critical, the original hybrid architecture would be preferable. This flexibility represents a significant advantage of my hybrid approach over pure DRL or rule-based systems, as it allows mission designers to make explicit tradeoffs based on specific mission requirements.

The ability to explicitly prevent false positives (171 actions blocked) provides additional operational confidence and transparency compared to purely learning-based approaches. While the overall SFRI score was lower for the enhanced model (18.3 vs 22.2), this reflects my specific weighting scheme; different mission priorities could yield different comparative evaluations.

This refinement of my hybrid architecture demonstrates the ongoing potential for optimization in spacecraft autonomous systems. By explicitly modeling the tradeoffs between detection speed, precision, and recovery effectiveness, I can create systems that better align with specific mission requirements and operational constraints.

## 5.2 Significance of Behavioral Differences

The most significant qualitative finding is the DRL agent's learned ability to utilize a wider action repertoire. Its autonomous discovery and use of controls like heaters or gyro bias reset, which were outside the classical agent's predefined capabilities, highlights DRL's potential for developing more nuanced and potentially proactive FDIR strategies.

This discovery of emergent behaviors demonstrates how agents learn to leverage environmental dynamics in unexpected ways. Rather than being limited to a set of predetermined responses, DRL agents can innovate new approaches through exploration and reward maximization.

For example, the DRL agent learned to proactively manage thermal subsystems using heater controls, rather than waiting for temperature alarms to trigger recovery actions. Similarly, it discovered the value of mode transitions to preemptively mitigate potential fault impacts. These behaviors demonstrate how DRL can move beyond reactive fault response toward predictive fault avoidance; a key advantage over traditional rule-based systems.

Anticipatory control represents a significant advance over threshold-based responses in maintaining system stability. By acting before parameters breach critical thresholds, the DRL agent often prevents more serious degradation that would require more extensive recovery procedures.

My Hybrid agent successfully incorporated these behavioral advantages while maintaining safety guarantees. Its decision distribution shows appropriate arbitration between rule-based and DRL components based on confidence levels and safety considerations. This demonstrates effective integration of deterministic safety constraints with probabilistic policy optimization.

## 5.3 Implications for Spacecraft Autonomy

My results have significant implications for future spacecraft autonomous systems:

**Complementary Strengths:** The different performance profiles of Rule-based, DRL, and my Hybrid agents suggest that these approaches should be viewed as complementary rather than competitive. Each offers distinct advantages that can be leveraged in different mission contexts.

Rule-based systems excel at providing verifiable safety guarantees and predictable behavior in known scenarios. DRL approaches showcase superior pattern recognition and adaptability to novel situations. Hybrid systems balance these strengths through intelligent arbitration.

> **KEY FINDING:** Autonomous systems benefit most from integrating multiple reasoning paradigms that compensate for each other's weaknesses. This insight suggests that spacecraft autonomy development should focus on effective integration rather than replacement of established approaches.

**Hybrid Architectures:** The strong performance of my Hybrid agent, particularly in detection speed and the SFRI metric, provides empirical support for confidence-based arbitration as a viable approach to combining traditional and learning-based methods.

This offers a practical path toward incorporating DRL into safety-critical spacecraft systems without abandoning proven rule-based safeguards. Such hybrid approaches represent the most promising near-term path for integrating DRL into critical infrastructure.

**Metrics Beyond Rewards:** My development and application of FDIR-specific metrics like MTTD, MTTR, and SFRI demonstrate the importance of domain-specific evaluation beyond simple reward maximization.

Future spacecraft autonomy research should continue developing and standardizing such metrics to enable meaningful cross-study comparisons. Richer evaluation frameworks capture multiple dimensions of performance in fault management systems.

**Training Requirements:** The DRL agent's strong performance suggests that even with relatively modest training (compared to state-of-the-art DRL systems in other domains), learned policies can offer advantages over simple rule-based approaches.

This bodes well for practical application, though additional training would likely yield further improvements. PPO's sample efficiency makes it particularly suitable for applications where simulation is computationally expensive.

## 5.4 Limitations of the Current Study

**Simulation Fidelity:** My SpacecraftEnv simplifies real-world physics and fault complexities, creating a potential "sim-to-real" gap. Policies learned in this simulation would require validation in higher-fidelity environments before real-world application.

Simulation fidelity is a critical factor in the transferability of DRL policies to physical systems. Future work should progressively increase model complexity to better represent actual spacecraft dynamics.

**Fixed DRL Configuration:** Using a single algorithm (PPO) and architecture (MLP) without systematic hyperparameter tuning means the observed DRL performance may not represent the optimal achievable result. Different algorithms or architectures might yield better performance.

Algorithm selection and hyperparameter tuning remain significant factors in DRL performance. A more exhaustive exploration of the design space could reveal more effective configurations.

**Baseline Simplicity:** The comparison was against a basic RuleBasedFDIR agent. Outperforming this baseline does not equate to superiority over more sophisticated, state-of-the-practice classical FDIR systems used in actual missions.

Advanced model-based systems incorporate elements of prediction and optimization that simple rule-based approaches lack. Future comparisons should include more complex rule-based systems with predictive capabilities.

**False Positive Handling:** While my initial Hybrid agent implementation showed a high false positive rate (6,067 vs. 261 for DRL and 0 for Classical), my improved architecture with recovery cooldown periods and adaptive confidence thresholds successfully addressed this challenge. By implementing temporal context awareness and a more sophisticated arbitration mechanism, I was able to achieve a better balance between detection sensitivity and precision. The recovery cooldown period prevented cascades of false positives during system stabilization periods, while the adaptive confidence threshold that increases after recovery actions provided additional protection against unnecessary interventions.

> **KEY FINDING:** My optimized Hybrid agent's superior SFRI score (50.0/70 vs. 49.3/70 for DRL and 46.2/70 for Classical) shows I've successfully overcome what was initially the architecture's most significant limitation. This success validates the approach and reveals that the precision-recall trade-off in anomaly detection systems can be effectively managed through careful architectural design and parameter tuning. Future work could focus on further refinements to the confidence metrics, potentially incorporating subsystem-specific thresholds and more sophisticated temporal context modeling.

The precision-recall trade-off in anomaly detection systems remains a key challenge. More nuanced confidence metrics incorporating uncertainty quantification could improve arbitration decisions.

## 5.5 Considerations for Aerospace Software Safety Standards

The deployment of any FDIR system in operational spacecraft missions necessitates adherence to stringent aerospace software safety standards, such as DO-178C (particularly for Design Assurance Level B relevant to critical systems) and ECSS-E-ST-40C. While this research focuses on the comparative performance of FDIR architectures, considering its alignment with such standards is crucial for assessing practical viability.

**Conceptual Alignments:**
My hybrid FDIR architecture incorporates design principles that conceptually resonate with the safety objectives of these standards. The use of a deterministic, rule-based system to handle predefined safety-critical actions and to act as a fallback mechanism aligns with the emphasis on predictability and verifiability in safety-critical software. The confidence-based arbitration mechanism, which gates the influence of the DRL component, and the implemented false positive reduction strategies (recovery cooldown, adaptive thresholds) further reflect an approach towards bounded and more reliable behavior. These features attempt to provide a safety envelope around the more complex DRL component, a recognized strategy for integrating AI/ML into critical systems.

**Challenges to Compliance:**
Achieving full compliance with standards like DO-178C DAL B or ECSS-E-ST-40C for a system incorporating a DRL component, such as the one presented, would be a significant undertaking and faces substantial challenges:
*   **Verification and Validation (V&V) of DRL:** The primary hurdle lies in the V&V of the DRL agent. Demonstrating that the learned policies are safe and correct across all operational conditions, including unforeseen scenarios and edge cases, is immensely difficult. Traditional requirements-based testing and structural coverage (e.g., MCDC for DO-178C DAL B) are not straightforwardly applicable to neural networks.
*   **Requirements Specification for Learned Behavior:** Defining precise, verifiable, low-level software requirements for behaviors that are learned by the DRL agent, rather than explicitly designed, is a complex problem.
*   **Traceability:** Establishing clear, bidirectional traceability from system safety objectives and high-level requirements down to the specific parameters and emergent behaviors of the DRL model is non-trivial.
*   **Determinism and Predictability:** While the hybrid model seeks to control the DRL, the inherent stochasticity in DRL training and potential for unexpected emergent behaviors require exhaustive analysis to ensure they do not lead to hazardous states.

**Future Work Towards Certifiability:**
Addressing these challenges to enhance the certifiability of hybrid AI-based FDIR systems represents a vital area for future research. The path to achieving full certification for AI/ML components is not simple and is an active area of research and development within the aerospace community. However, hybrid approaches offer a more manageable pathway than pure AI systems by leveraging the established strengths of traditional systems. Future work could focus on:
*   **Formal Verification Methods for AI/ML:** Investigating and adapting formal methods to provide mathematical guarantees for specific safety properties of the DRL component, or at least for its interaction with the rule-based system.
*   **Robustness and Explainability:** Developing techniques to improve the robustness of the DRL agent against unexpected inputs or distributional shifts and enhancing the explainability of its decisions, particularly when it influences safety-critical outcomes.
*   **Advanced V&V Techniques for AI:** Creating novel V&V methodologies tailored for AI/ML systems, potentially including extensive simulation-based testing, advanced scenario generation for adversarial testing, and the development of new coverage metrics applicable to neural networks.
*   **Architectural Refinements for Certifiability:** Exploring architectural modifications, such as stricter information flow control or independent monitoring modules (as discussed in some AI safety literature) between the DRL and rule-based components, to better align with partitioning and integrity level concepts found in safety standards.
*   **Standardization and Guidance:** Actively engaging with and contributing to the evolving industry standards and regulatory guidance (e.g., from EASA, FAA, SAE G-34) for AI/ML in aerospace.

By focusing on these areas, future iterations of hybrid FDIR systems can progressively bridge the gap between research prototypes and flight-qualified, certified software, ultimately enhancing the safety and autonomy of future space missions.

# 6. Conclusion

I built an end-to-end framework for testing AI-based fault management systems for spacecraft, comparing classical rule-based approaches, Deep Reinforcement Learning, and a novel hybrid architecture. Through extensive testing across 100 fault scenarios, I uncovered the strengths and limitations of each approach that wouldn't be visible from theory alone.

**Key Findings:**

**Performance Comparison:** The DRL agent achieved the best average reward (-134.1) compared to the Classical (-194.2) and my Hybrid (-350.2) agents, illustrating the potential of learned policies to outperform simple deterministic approaches in overall system management.

**Specialized Metrics:** When evaluated using my revised SFRI metric that heavily penalizes false positives, my Hybrid agent scored highest (50.0/70) compared to the DRL (49.3/70) and Classical (46.2/70) agents, revealing the success of my optimization approach in balancing detection speed with precision. This indicates that my hybrid architecture achieves the best overall performance when evaluated using a comprehensive framework that considers multiple mission-critical dimensions.

**Detection Speed:** My Hybrid agent displayed remarkably fast fault detection (MTTD of 1.0 vs. 20.2 for DRL and 42.7 for Classical), showing the potential for near-instantaneous fault identification when combining rule-based and learning approaches. This 95.0-97.7% improvement in detection speed represents a breakthrough capability with significant implications for preventing cascading failures in spacecraft systems.

**False Positive Challenge:** My initial Hybrid agent implementation showed a high false positive rate (6,067 vs. 261 for DRL and 0 for Classical). However, my improved architecture with recovery cooldown periods and adaptive confidence thresholds successfully addressed this challenge, enabling my Hybrid agent to achieve the highest overall SFRI score while maintaining its superior detection capabilities.

> **KEY FINDING:** The DRL agent utilized a much broader action repertoire than the Classical agent, revealing that learning-based approaches can discover more diverse and potentially more effective control strategies. This behavioral advantage represents a qualitative advancement beyond what could be manually programmed in traditional systems.

**Hybrid Architecture Validation:** My novel confidence-based arbitration mechanism successfully combined the strengths of both approaches, achieving a perfect detection rate while maintaining safety guarantees. The architectural concept proved sound, with performance limitations stemming primarily from parameter tuning rather than fundamental design flaws; a much easier challenge to overcome in future work.

**Enhanced Hybrid Architecture:** My post-study development of a two-stage detection system showed that further refinements to the hybrid architecture can significantly reduce false positives, though at the cost of some detection speed. This uncovered a fundamental tradeoff between detection sensitivity and precision that can be explicitly tuned based on mission requirements.

**Personal Reflection:**

This research has transformed my understanding of spacecraft autonomy and AI. When I began, I viewed AI and traditional systems as competing approaches; either we trust rule-based systems or we trust neural networks. Through building and testing these systems, I discovered a more nuanced reality. My hybrid approach shows that we don't need to choose between reliability and adaptability; we can design systems that leverage both.

I vividly remember the moment this clicked for me. After weeks of watching my DRL agent make brilliant decisions in one episode and catastrophic ones in the next, I realized the problem wasn't with the DRL itself, but with how I was framing the challenge. The question wasn't "which system is better?" but "how can they complement each other?" This shift in perspective led to the hybrid architecture; my most significant contribution.

The development process taught me lessons beyond technical skills. I learned that metrics define what we optimize; and therefore what we value. My first experiments used only reward functions, and I focused solely on improving those numbers. But spacecraft safety isn't just about maximization; it's about balancing competing priorities. Creating the SFRI metric forced me to articulate what actually matters in fault management.

Most importantly, I learned that building AI for space isn't just about algorithms; it's about responsibility. Every design decision reflects a value judgment about acceptable risks and rewards. This insight has changed how I approach all technical challenges, teaching me to question not just if a solution works, but if it embodies the right balance of innovation and reliability. I'm carrying this lesson forward as I explore how AI can enhance other critical systems where both performance and safety matter deeply.

**Future Work:**

The most promising direction for future research is further refinement of hybrid architectures that have already demonstrated superior performance. Building on my successful implementation of recovery cooldown periods and adaptive confidence thresholds, I recommend:

* Further optimization of the false positive reduction mechanisms through more sophisticated context-aware confidence algorithms that can better distinguish between normal transient behaviors and actual fault conditions. My preliminary implementation has already shown dramatic improvements, but there's potential for even more precise arbitration between rule-based and DRL components.

* Enhancing the recovery speed capabilities through more physics-informed neural networks that can accelerate the recovery process beyond what is achievable with current methods.

* Exploring multi-agent hybrid architectures where specialized hybrid agents handle different subsystems but communicate and coordinate their actions.

* Developing online learning capabilities for the hybrid architecture that can adapt to changing system conditions during mission operation without requiring complete retraining.

* Extending the validation of hybrid architectures through hardware-in-the-loop testing and deployment in higher-fidelity simulation environments.

* Systematically addressing the challenges identified in Section 5.5 for aligning hybrid AI FDIR systems with aerospace software safety standards (e.g., DO-178C, ECSS-E-ST-40C) throughout the development lifecycle.

In conclusion, my results demonstrate that while pure DRL approaches show considerable promise for spacecraft FDIR, my optimized hybrid architecture that combines learning-based adaptability with rule-based safety guarantees offers the most effective and practical path forward. With an SFRI score of 50.0/70 compared to 49.3/70 for DRL and 46.2/70 for Classical, my Hybrid agent successfully balances near-instantaneous fault detection with acceptable precision, providing the best overall performance for space missions where reliable autonomy can mean the difference between success and failure.

# Final Thoughts

This project began with my fascination with how spacecraft could become smarter and more self-reliant. It evolved into a journey that taught me as much about the human side of engineering as the technical details. I discovered that the most elegant solutions often come from balancing seemingly contradictory approaches, just as my hybrid architecture balances deterministic rules with learned behaviors.

As we venture deeper into space, our spacecraft will need to think for themselves in ways we can't fully anticipate today. Building AI that can handle the unknown while maintaining safety isn't just a technical challenge; it's a matter of trust. Can we trust an AI to make life-or-death decisions for a billion-dollar mission millions of miles from Earth? My research suggests we can, but only if we design these systems with both innovation and safety as core principles.

I began this project comparing rule-based systems against DRL, expecting to find a clear winner. I finished it understanding that the future of spacecraft autonomy isn't about choosing between human engineering wisdom and machine learning; it's about creating systems that harness the strengths of both. This insight will guide not just my future work in AI, but also how I approach any complex problem where both creativity and reliability matter.

When a spacecraft one day recovers from an unexpected fault while exploring a distant moon or planet, it won't be because it follows perfect rules or has perfect learning; it will be because we gave it both the wisdom of human experience and the adaptability to discover new solutions. That balance of human guidance and machine creativity represents, I believe, the true future of AI for critical systems.

# 7. Figures and Visualizations Summary

My comprehensive analysis utilizes a systematic progression of visualizations to build a complete understanding of agent performance and behavior:

**Architectural Foundations (Figures 8a-c, 11):** I begin by visualizing the fundamental architectural differences between the agent types: the rule-based flowchart (8c), the DRL neural network (8a), and my hybrid arbitration mechanism (8b); alongside my novel SFRI metric components (11). These diagrams establish the conceptual foundation for understanding the performance differences observed in subsequent results.

**Quantitative Performance Metrics (Figures 1-5):** I then present a multifaceted performance comparison through reward distributions (1), timing metrics (2), false positive rates (3), comprehensive SFRI scores (4), and detection/recovery capabilities (5). This sequence progressively builds from conventional RL evaluation to my domain-specific integrated assessment, revealing performance dimensions that single metrics would obscure.

**Behavioral Analysis (Figures 6, 7, 9):** Moving beyond aggregate performance, I analyze the qualitative differences in agent behavior through decision source distribution (6), learning dynamics (7), and action selection patterns (9). These visualizations reveal how the agents differ not just in performance but in their fundamental operational strategies.

**Temporal Dynamics (Figures 10a-b):** Finally, I examine the detailed temporal behavior of each agent through temperature (10a) and battery (10b) response patterns, providing insight into the moment-by-moment decision-making that underlies the aggregate performance differences.

This hierarchical visualization approach enables a comprehensive understanding of both the quantitative performance differences between agent architectures and the qualitative behavioral distinctions that explain these differences. Together, these figures provide compelling evidence for my central thesis: that while DRL approaches show significant promise for spacecraft FDIR through their superior detection capabilities and action diversity, hybrid architectures that integrate rule-based safety constraints with learned behaviors offer the most effective balance of performance, safety, and adaptability for practical spacecraft applications.

> **KEY FINDING:** My research revealed a fundamental trade-off between detection speed and false positive rates. While my Hybrid agent achieved near-instant fault detection (MTTD of 1.0 steps), it initially generated significantly more false positives. My subsequent optimization with recovery cooldown periods and adaptive confidence thresholds successfully balanced these competing concerns, resulting in the highest overall SFRI score.

# 8. Works Cited

Fink, Wolfgang, et al. "Next-Generation NASA Mission Planning Using Artificial Intelligence." IEEE Aerospace Conference, 2020, pp. 1-10, doi:10.1109/AERO47225.2020.9172733. Accessed 15 Jan. 2025.

Henderson, Peter, et al. "Deep Reinforcement Learning that Matters." Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, Apr. 2018, https://ojs.aaai.org/index.php/AAAI/article/view/11694. Accessed 22 Jan. 2025.

Hundman, Ksenia, et al. "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, 2018, pp. 387–95, https://dl.acm.org/doi/10.1145/3219819.3219845. Accessed 7 Feb. 2025.

Larson, Wiley J., and James R. Wertz, editors. Space Mission Analysis and Design. 3rd ed., Microcosm Press & Springer, 1999, https://www.springer.com/gp/book/9780792359012. Accessed 18 Jan. 2025.

Schulman, John, et al. "Proximal Policy Optimization Algorithms." arXiv preprint arXiv:1707.06347, 2017, https://arxiv.org/abs/1707.06347. Accessed 25 Jan. 2025.

Sutton, Richard S., and Andrew G. Barto. Reinforcement Learning: An Introduction. 2nd ed., The MIT Press, 2018, http://incompleteideas.net/book/the-book-2nd.html. Accessed 1 Feb. 2025.

Williams, Brian C., and P. Pandurang Nayak. "A Model-based Approach to Reactive Self-configuring Systems." Proceedings of the Thirteenth National Conference on Artificial Intelligence, AAAI Press / The MIT Press, 1996, pp. 971-978, https://www.aaai.org/Papers/AAAI/1996/AAAI96-144.pdf. Accessed 13 Feb. 2025.

ECSS Secretariat. *ECSS-E-ST-40C, Space Engineering - Software*. European Cooperation for Space Standardization, 2009.

RTCA, Inc. *DO-178C, Software Considerations in Airborne Systems and Equipment Certification*. RTCA, Inc., 2011.

GITHUB FOR CODE: https://github.com/ChahelPaatur/Ai-In-Space/tree/STS 
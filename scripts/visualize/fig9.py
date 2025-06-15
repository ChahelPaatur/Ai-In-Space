import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sample data structure (replace with your actual data)
action_types = ['No-op', 'RecoverEPS', 'RecoverADCS', 'RecoverTCS', 
                'HeaterON', 'HeaterOFF', 'ResetGyroBias', 'EnterSafe', 'EnterNominal']

# Create a dictionary with your actual action frequencies for each agent
data = {
    'Action': action_types * 3,
    'Frequency': [
        # Classical agent frequencies (example values)
        0.50, 0.25, 0.15, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00,
        # DRL agent frequencies
        0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.05, 0.05,
        # Hybrid agent frequencies
        0.40, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05
    ],
    'Agent': ['Classical'] * 9 + ['DRL'] * 9 + ['Hybrid'] * 9
}

# Create DataFrame
df = pd.DataFrame(data)

# Create the plot
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Action', y='Frequency', hue='Agent', data=df)
plt.title('Figure 9: Action Distribution Across Agent Types (n=100 episodes)')
plt.xlabel('Action Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.legend(title='Agent Type')
plt.tight_layout()
plt.savefig('figure9_action_distribution.png', dpi=300)
plt.show()

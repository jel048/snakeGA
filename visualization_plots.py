import pandas as pd
import matplotlib.pyplot as plt

version = 'v15.1'

#Load the CSV file into a DataFrame
df =pd.read_csv(f'model_logs/{version}.csv')

# Plot 1: Length Mean vs Iteration
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['length_mean'], color='b', label='Length Mean')
plt.xlabel('Iteration')
plt.ylabel('Length Mean')
plt.title('Iteration vs Length Mean')
plt.grid(True)
plt.legend()
plt.savefig(f'images/length_mean_vs_iteration_{version}.png')  # Save the plot as an image
plt.clf()  # Clear the current figure for the next plot

# Plot 2: Reward Mean vs Iteration
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['reward_mean'], color='g', label='Reward Mean')
plt.xlabel('Iteration')
plt.ylabel('Reward Mean')
plt.title('Iteration vs Reward Mean')
plt.grid(True)
plt.legend()
plt.savefig(f'images/reward_mean_vs_iteration_{version}.png')  # Save the plot as an image

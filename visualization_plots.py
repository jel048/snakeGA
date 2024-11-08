import pandas as pd
import matplotlib.pyplot as plt

version = 'v17.1'

#Load the CSV file into a DataFrame
df =pd.read_csv(f'model_logs/{version}.csv')

#Plot 1: Length Mean vs Iteration
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['length_mean'], color='b', label='Length Mean')
plt.xlabel('Iteration')
plt.ylabel('Length Mean')
plt.title(f'Mean length of snake during training, version: {version}')
plt.grid(True)
plt.legend()
plt.savefig(f'images/length_mean_vs_iteration_{version}.png')  #save the plot as an image
plt.clf()  #clear the figure for the next plot

#Plot 2: Reward Mean vs Iteration
plt.figure(figsize=(10, 6))
plt.plot(df['iteration'], df['reward_mean'], color='g', label='Reward Mean')
plt.xlabel('Iteration')
plt.ylabel('Reward Mean')
plt.title(f'Mean rewards during training, version: {version}')
plt.grid(True)
plt.legend()
plt.savefig(f'images/reward_mean_vs_iteration_{version}.png')  #saave the plot as an image

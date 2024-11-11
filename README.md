# snakeGA

## DTE2502 – Neural Networks :: Graded Assignment 02

# Runnng Graded assignment 02

## To run the project, follow these steps:  

Clone the GitHub repository and navigate to the main directory.  
Ensure that all dependencies match those specified in the provided environment.yaml file.  
set the version variable in training.py to the model you want to train. Example : "v15.1"  
Execute training.py to begin the training process.  
To visualize gameplay results, run game_visualization.py, after setting model version also here.  


## Overview

This project implements a Deep Reinforcement Learning agent to play the game of Snake, utilizing Deep Q-Learning with convolutional neural networks. This code is a PyTorch adaptation based on the snake-rl project by [DragonWarrior15](https://github.com/DragonWarrior15/snake-rl). 
The main goal of the assignment was to convert the DeepQLearningAgent from TensorFlow to Pytorch. 
Following the changes to this class also led to small changes in other parts of the codebase, to ensure every part is compatible with the new pytorch code.

My other goals include training an agent capable of navigating an obstacle-free board and a more challenging board with randomly generated obstacles.


## Sample Gameplay  
Below are sample games from the best-performing models:

## Version 15 - Obstacle-Free Environment
Trained in an open environment without obstacles, the version 15 agent demonstrates the agent in a classic Snake game setup.  
This model is trained with two convolution layers, one with 16 filters [3,3] kernel size, and one with 32 filters and [3,3] kernel size  

Average game rewards and average snake length during training:  

***
<img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/reward_mean_vs_iteration_v15.1.png" alt="model v15.1 reward mean" ><img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/length_mean_vs_iteration_v15.1.png" alt="model v15.1 length mean >

***


Sample games from one of the best performing version 15 [agent](..\models\v15.1\model_193500.pth)<br>  

***
<img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v15.1.2.gif" alt="model v15.1 agent, gif #2" ><img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v15.1.3.gif" alt="model v15.1 agent, gif 3" >
***

## Version 16 - Obstacle-Free Environment, extra CNN layer
Trained in an open environment without obstacles, but with one more convolution layer than version 15, also this one with 32 filters. The version 16 agent demonstrates the increased learning of an extra layer.

Average game rewards and average snake length during training:

***
<img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/reward_mean_vs_iteration_v16.1.png" alt="model v16.1 reward mean" ><img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/length_mean_vs_iteration_v16.1.png" alt="model v16.1 length mean >

***

Sample games from one of the best performing version 16 [agent](..\models\v16.1\model_197500.pth)<br>  

***
<img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v16.1.1.gif" alt="model v116.1 agent, gif #1" ><img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v16.1.3.gif" alt="model v16.1 agent, gif 3" >
***

## Version 17 - Environment with Obstacles
In this setup, obstacles are randomly generated on the board. Version 17 of the model generalizes well to the complex environment.

Average game rewards and average snake length during training:

***
<img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/reward_mean_vs_iteration_v17.1.png" alt="model v17.1 reward mean" ><img width="400" height="200" src="https://github.com/jel048/snakeGA/blob/master/images/length_mean_vs_iteration_v17.1.png" alt="model v17.1 length mean >

***

Sample games from one of the best performing version 17 [agent](..\models\v17.1\model_193000.pth)<br>  
***
<img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v17.1.3.gif" alt="model v17.1 agent, gif #3" ><img width="400" height="400" src="https://github.com/jel048/snakeGA/blob/master/images/v17.1.2.gif" alt="model v17.1 agent, gif 2" >
***



## Code Structure
game_environment.py: Contains the Snake environment class (SnakeNumpy), with an interface similar to OpenAI Gym for easy interaction and simulation.  

agent.py: Implements the DeepQLearningAgent using PyTorch. A JSON config file under model_config is read to set up the convolutional neural network according to specifications in the file.
the DeepQLearningAgent uses a target net to keep weights stable and to avoid chasing a moving target. 


training.py: Manages the training process, integrating environment interactions with agent learning and model updates.
log_frequency, by default set to 500, determins how often to update the target net with current model weights, save a copy of the model, and update the log files with progress.

game_visualization.py: Converts gameplay to an mp4 format for visual analysis.


## Training Methodology
The agent first plays a series of games to fill the replay buffer. Then, all training happens by sampling random batches from this buffer.  
The Deep Q-Learning agent in this project uses two frames as input to predict action values, supporting decisions for the next move.  
As the training goes on, new games are played to fill the replay buffer with better games.  
The model trains using its model network, coupled with a target network, which is updated periodically with the models own weights.


## Optimizer and loss
Optimizer: RMSProp  
Adaptive learning rate optimizer designed to handle non-stationary objectives by keeping a moving average of squared gradients.  <br>

Loss: Huber Loss  
Loss function that combines Mean Squared Error (MSE) and Mean Absolute Error (MAE), balancing sensitivity to outliers and convergence stability.  
It behaves like MSE for small errors, but reduces the impact of large errors by acting like MAE beyond a certain threshold, often used in reinforcement learning.


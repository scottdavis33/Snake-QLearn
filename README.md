# Snake-QLearn

## Overview
This project implements a Q-Learning model for the classic game Snake. The goal is to optimize the snake's ability to survive and score high by learning through reinforcement learning techniques.

## Current State of the Model

### Model Architecture
- **Loss Function:** Mean Squared Error (MSE)
- **Input Features:** 12
- **Output Features:** 3
- **Hidden Layers:** 1 hidden layer with 256 neurons

### Rewards System
- **Food Reward:** +10 for getting food
- **Efficiency Penalty:** -0.01 multiplied by the number of steps taken since the last food was eaten (Might want to try a smaller value here)
- **Collision Penalty:** -10 for colliding with the walls or itself
- **Score Milestone Reward:** +5 if the score is a multiple of 15

### State Representation
The state vector provided to the model includes:
- **Danger Left:** Presence of immediate danger to the left
- **Danger Right:** Presence of immediate danger to the right
- **Danger Straight:** Presence of immediate danger straight ahead
- **Direction:** Current direction of the snake
- **Food Location:** Relative position of food compared to the snake’s head
- **Length of Snake:** Current length of the snake

## Future Directions
- **Explore different sizes for the hidden layer:** Experiment with various sizes to find the optimal configuration.
- **Adjust reward/punishment criteria:** Test different schemes to enhance learning efficiency and performance.


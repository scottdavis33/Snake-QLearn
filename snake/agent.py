import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_Qnet, QTrainer
from helper import plot
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 100
MAX_GAMES = 20
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        
        self.epsilon = 1.0  # Starting value of epsilon
        self.epsilon_min = 0  # Minimum value of epsilon
        self.epsilon_decay = 0.999  # Decay rate per game
        
        self.gamma = 0.9 # discount rate 
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_Qnet(12, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma = self.gamma)

    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
            
            
            # Length of Snake
            len(game.snake)  # Adding snake length to the state
            
            ]
        
        

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        # states, actions, rewards, next_states, dones = zip(*mini_sample)    
        # self.trainer.train_step(states, actions, rewards, next_states, dones)
        for state, action, reward, next_state, done in mini_sample:
           loss = self.trainer.train_step(state, action, reward, next_state, done)
        return loss
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
         # Apply exponential decay to epsilon
         
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        final_move = [0, 0, 0]
        
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
                
def train(max_games = None):
    
    plot_scores = []
    plot_mean_scores = []
    plot_records = []
    plot_losses = []
    total_score = 0
    mean_loss = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    try:
        while True:
            # get old (curr) state 
            state_old = agent.get_state(game)
            
            # get move
            final_move = agent.get_action(state_old)
            
            # perform move and get new state
            
            reward, done, score = game.play_step(final_move)
            
            state_new = agent.get_state(game)
            
            # train short memory 
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            
            
            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
            
            if done:
                # train the long memory, plot results
                game.reset()
                agent.n_games += 1
                loss = agent.train_long_memory()
                # agent.train_long_memory()
                
                plot_losses.append(loss)
                average_loss = np.mean(plot_losses)
                
                if score > record:
                    record = score
                    agent.model.save()
                    
                total_score += score
                mean_score = total_score / agent.n_games
                
                print('Game:', agent.n_games, 'Score:', score, 'Record:', record, 'Average Score:', np.round(mean_score, 4), 'Average Loss:', np.round(average_loss,4), 'Epsilon', agent.epsilon)


                plot_records.append(record)
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
                
                if max_games and agent.n_games >= max_games:
                    break  # Break after reaching the max number of games
                
    except KeyboardInterrupt:
        print('Training interrupted by user.')
    finally:
        print('Training completed.')
        return plot_scores, plot_mean_scores, plot_records
            
                    
if __name__ == '__main__':
    train()

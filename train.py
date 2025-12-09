import pygame
import os
import torch
from agent import DQN, DQNAgent
from environment import Environment

# Choosing the device either CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The size of the map 
n = 14 

# The size of the PyGame screen and screen itself
scrx = n*50
scry = n*50
screen = pygame.display.set_mode((scrx,scry)) 

# The size of the state, action and value size
state_size = n # n*n
action_size = n # n*n
value_size = 2

# The batch size used to sample the replay memory to train the agent 
batch_size = 16

# The number of episodes for an RL algo
n_episodes = 1000

# The initial walkable tiles of the first map layout
initial_walkable_tiles = 60

# Creating an output folder for the model weights
dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path ,"model_output/dqn/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initializing the model and agent
model = DQN(state_size, action_size, value_size)
model.to(device)
agent = DQNAgent(state_size, action_size, value_size, model, device) 

# Initializing the environment
env = Environment(state_size, action_size, value_size, scrx, scry, screen, initial_walkable_tiles)
map_layout = env.reset()

# The main training loop for the the DQN algorithme
run = True
while run:
    done = 0
    for e in range(n_episodes):
        map_layout = env.reset()
        for frame in range(500):
            env.render()
            pygame.display.flip()
            action, value = agent.act(torch.tensor(map_layout, dtype=torch.float).unsqueeze(0).unsqueeze(0))
            new_map_layout, reward, done = env.step(action, value)
            # if the 500 frames weren't enough
            reward = reward if not done else -10
            agent.remember(torch.tensor(map_layout, dtype=torch.float).unsqueeze(0).unsqueeze(0),\
                torch.tensor(action, dtype=torch.int8).unsqueeze(0), \
                torch.tensor(reward).unsqueeze(0), \
                torch.tensor(value, dtype=torch.int8).unsqueeze(0),\
                torch.tensor(new_map_layout, dtype=torch.float).unsqueeze(0).unsqueeze(0), \
                torch.tensor(done, dtype=torch.int8).unsqueeze(0))
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, frame, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if e % 50 == 0:
                agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".pt")   

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
    
pygame.quit()
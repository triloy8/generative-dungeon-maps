import os

import pygame
import torch
from tqdm import tqdm

from agent import DQNAgent
from environment import Environment

try:
    import wandb
except ImportError:
    wandb = None

# Choosing the device either CPU or GPU / dtype
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

# The size of the map
n = 7

# The initial walkable tiles of the first map layout
initial_walkable_tiles = 24

# The size of the PyGame screen and screen itself
scrx = n*50
scry = n*50
screen = pygame.display.set_mode((scrx,scry)) 

# The size of the state, action and value size
state_size = n # n*n
action_size = n # n*n
value_size = 2

# The batch size used to sample the replay mesmory to train the agent 
batch_size = 16

# The number of episodes for an RL algo / number of frames 
n_episodes = 1000
n_frames = 100

# Target path boundary
target_path = 0

# Creating an output folder for the model weights
dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path ,"model_output/dqn/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initializing agent
agent = DQNAgent(state_size, action_size, value_size, dtype, device)

wandb_run = None
if wandb is not None:
    wandb_run = wandb.init(
        project="dqn-debug",
        config={
            "n": n,
            "state_size": state_size,
            "batch_size": batch_size,
            "n_episodes": n_episodes,
            "n_frames": n_frames,
            "initial_walkable_tiles": initial_walkable_tiles,
            "target_path": target_path,
            "epsilon_decay": agent.epsilon_decay,
            "learning_rate": agent.learning_rate,
        },
    )

# Initializing the environment
env = Environment(state_size, action_size, value_size, scrx, scry, screen, initial_walkable_tiles, target_path)
map_layout = env.reset()

# The main training loop for the the DQN algorithm
global_step = 0
while True:
    done = 0
    for e in tqdm(range(n_episodes), desc="Number of episodes", unit="episode"):
        map_layout = env.reset()
        episode_reward = 0.0
        for frame in tqdm(range(n_frames), desc="Number of frames", unit="frame"):
            env.render()
            pygame.display.flip()
            action, value = agent.act(torch.tensor(map_layout, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0))
            new_map_layout, reward, done = env.step(action, value)
            episode_reward += reward
            agent.remember(
                state=torch.tensor(map_layout, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0),
                action=torch.tensor(action, dtype=torch.int8, device=device).unsqueeze(0),
                value=torch.tensor(value, dtype=torch.int8, device=device).unsqueeze(0),
                reward=torch.tensor(reward, dtype=dtype, device=device).unsqueeze(0),
                next_state=torch.tensor(new_map_layout, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0),
                done=torch.tensor(done, dtype=torch.int8, device=device).unsqueeze(0),
            )
            map_layout = new_map_layout
            global_step += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "reward": reward,
                        "epsilon": agent.epsilon,
                        "done_flag": float(done),
                        "frame_index": frame,
                        "episode_index": e,
                    },
                    step=global_step,
                )
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes, frame, agent.epsilon))
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "episode_reward": episode_reward,
                            "episode_length": frame + 1,
                            "episode_index": e,
                        },
                        step=global_step,
                    )
                break
            replay_loss = None
            replay_stats = None
            if len(agent.memory) >= batch_size:
                replay_loss, replay_stats = agent.replay(batch_size)
            if wandb_run is not None and replay_loss is not None:
                log_payload = {"replay_loss": replay_loss}
                if replay_stats:
                    log_payload.update(replay_stats)
                wandb_run.log(log_payload, step=global_step)

        if e % 50 == 0:
            agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".pt")
            if wandb_run is not None:
                wandb_run.log({"checkpoint_episode": e}, step=global_step)

            
    
pygame.quit()
if wandb_run is not None:
    wandb_run.finish()
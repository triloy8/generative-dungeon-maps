import os

import numpy as np
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


def observation_to_tensor(observation):
    stacked = np.stack(
        [
            observation["map"].astype(np.float32),
            observation["heatmap"].astype(np.float32),
        ],
        axis=0,
    )
    return torch.tensor(stacked, dtype=dtype, device=device).unsqueeze(0)

# The size of the map
n = 14

# The size of the PyGame screen and screen itself
scrx = n*50
scry = n*50
screen = pygame.display.set_mode((scrx,scry)) 

# Grid size and value size
grid_size = n
value_size = 2

# The batch size used to sample the replay mesmory to train the agent 
batch_size = 16

# The number of episodes for an RL algo
n_episodes = 1000

# Target path boundary
target_path = 5

# Creating an output folder for the model weights
dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(dir_path ,"model_output/dqn/")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initializing the environment
env = Environment(grid_size, value_size, scrx, scry, screen, target_path)
env.reset()
frames_per_episode = env.max_iterations

# Initializing agent
agent = DQNAgent(grid_size, value_size, dtype, device)

wandb_run = None
if wandb is not None:
    wandb_run = wandb.init(
        project="dqn-debug",
        config={
            "n": n,
            "grid_size": grid_size,
            "batch_size": batch_size,
            "n_episodes": n_episodes,
            "n_frames": frames_per_episode,
            "target_path": target_path,
            "epsilon_decay": agent.epsilon_decay,
            "learning_rate": agent.learning_rate,
        },
    )

# The main training loop for the the DQN algorithm
run = True
global_step = 0
for e in tqdm(range(n_episodes), desc="Number of episodes", unit="episode"):
    if not run:
        break
    observation = env.reset()
    frames_per_episode = env.max_iterations
    episode_reward = 0.0
    state_tensor = observation_to_tensor(observation)
    for frame in tqdm(range(frames_per_episode), desc="Number of frames", unit="frame"):
        env.render()
        pygame.display.flip()
        action, value = agent.act(state_tensor)
        next_observation, reward, done = env.step(action, value)
        next_state_tensor = observation_to_tensor(next_observation)
        episode_reward += reward
        agent.remember(
            state=state_tensor,
            action=torch.tensor(action, dtype=torch.int8, device=device).unsqueeze(0),
            value=torch.tensor(value, dtype=torch.int8, device=device).unsqueeze(0),
            reward=torch.tensor(reward, dtype=dtype, device=device).unsqueeze(0),
            next_state=next_state_tensor,
            done=torch.tensor(done, dtype=torch.int8, device=device).unsqueeze(0),
        )
        state_tensor = next_state_tensor
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        if not run:
            break

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".pt")
        if wandb_run is not None:
            wandb_run.log({"checkpoint_episode": e}, step=global_step)

    if not run:
        break

pygame.quit()
if wandb_run is not None:
    wandb_run.finish()
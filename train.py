import argparse
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


device = None
dtype = None


def resolve_device(arg):
    if arg.lower() == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(arg)


def resolve_dtype(arg):
    mapping = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float64': torch.float64,
    }
    if arg not in mapping:
        raise ValueError(f"Unsupported dtype '{arg}'. Choose from {list(mapping.keys())}.")
    return mapping[arg]


def observation_to_tensor(observation):
    stacked = np.stack(
        [
            observation['map'].astype(np.float32),
            observation['heatmap'].astype(np.float32),
        ],
        axis=0,
    )
    return torch.tensor(stacked, dtype=dtype, device=device).unsqueeze(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Train the DQN level generator.')
    parser.add_argument('--map-size', type=int, default=7, help='Grid width/height.')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes.')
    parser.add_argument('--batch-size', type=int, default=16, help='Replay batch size.')
    parser.add_argument('--target-path', type=int, default=5, help='Required path improvement to finish.')
    parser.add_argument('--checkpoint-dir', default='model_output/dqn', help='Directory for model checkpoints.')
    parser.add_argument('--save-every', type=int, default=50, help='Episodes between checkpoints.')
    parser.add_argument('--render', action='store_true', help='Render pygame window during training.')
    parser.add_argument('--enable-wandb', action='store_true', help='Log metrics to Weights & Biases.')
    parser.add_argument('--project', default='dqn-debug', help='W&B project name.')
    parser.add_argument('--prob-empty', type=float, default=0.5, help='Initial probability for empty tiles.')
    parser.add_argument('--change-percentage', type=float, default=0.2, help='Fraction of tiles allowed to change.')
    parser.add_argument('--memory-capacity', type=int, default=10000, help='Replay memory capacity.')
    parser.add_argument('--gamma', type=float, default=0.95, help='Discount factor.')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help='Initial epsilon for exploration.')
    parser.add_argument('--epsilon-decay', type=float, default=0.999, help='Multiplicative epsilon decay per step.')
    parser.add_argument('--epsilon-min', type=float, default=0.01, help='Minimum epsilon value.')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Optimizer learning rate.')
    parser.add_argument('--clip-min', type=float, default=-10.0, help='TD target lower clamp.')
    parser.add_argument('--clip-max', type=float, default=10.0, help='TD target upper clamp.')
    parser.add_argument('--target-update-interval', type=int, default=2000, help='Steps between target network syncs.')
    parser.add_argument('--device', default='auto', help="Torch device to use ('auto', 'cpu', 'cuda', etc.).")
    parser.add_argument('--dtype', default='float32', help="Torch dtype (float32, float16, bfloat16, float64).")
    return parser.parse_args()


def train(args):
    global device, dtype
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    pygame.init()
    tile_size = 50
    scrx = args.map_size * tile_size
    scry = args.map_size * tile_size
    if args.render:
        screen = pygame.display.set_mode((scrx, scry))
    else:
        screen = pygame.Surface((scrx, scry))

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    env = Environment(
        args.map_size,
        2,
        scrx,
        scry,
        screen,
        args.target_path,
        prob_empty=args.prob_empty,
        change_percentage=args.change_percentage,
    )
    env.reset()
    agent = DQNAgent(
        args.map_size,
        2,
        dtype,
        device,
        memory_capacity=args.memory_capacity,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        learning_rate=args.learning_rate,
        clip_min=args.clip_min,
        clip_max=args.clip_max,
        target_update_interval=args.target_update_interval,
    )

    wandb_run = None
    if args.enable_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.project,
            config={
                'map_size': args.map_size,
                'batch_size': args.batch_size,
                'episodes': args.episodes,
                'target_path': args.target_path,
                'epsilon_start': args.epsilon_start,
                'epsilon_decay': args.epsilon_decay,
                'epsilon_min': args.epsilon_min,
                'gamma': args.gamma,
                'learning_rate': args.learning_rate,
                'memory_capacity': args.memory_capacity,
                'clip_min': args.clip_min,
                'clip_max': args.clip_max,
                'target_update_interval': args.target_update_interval,
                'prob_empty': args.prob_empty,
                'change_percentage': args.change_percentage,
            },
        )

    run = True
    global_step = 0
    for episode in tqdm(range(args.episodes), desc='Training episodes', unit='episode'):
        if not run:
            break
        observation = env.reset()
        frames_per_episode = env.max_iterations
        episode_reward = 0.0
        state_tensor = observation_to_tensor(observation)
        frame = -1
        frame_iter = tqdm(range(frames_per_episode), desc=f'Episode {episode + 1} frames', unit='frame')
        for frame in frame_iter:
            if args.render:
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
                        'reward': reward,
                        'epsilon': agent.epsilon,
                        'done_flag': float(done),
                        'frame_index': frame,
                        'episode_index': episode,
                    },
                    step=global_step,
                )

            if done:
                break

            if len(agent.memory) >= args.batch_size:
                replay_loss, replay_stats = agent.replay(args.batch_size)
                if wandb_run is not None:
                    payload = {'replay_loss': replay_loss}
                    if replay_stats:
                        payload.update(replay_stats)
                    wandb_run.log(payload, step=global_step)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break
            if not run:
                break

        if wandb_run is not None:
            wandb_run.log(
                {
                    'episode_reward': episode_reward,
                    'episode_length': frame + 1,
                    'episode_index': episode,
                },
                step=global_step,
            )

        if (episode + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'weights_{episode + 1:04d}.safetensors',
            )
            agent.save(checkpoint_path)
            if wandb_run is not None:
                wandb_run.log({'checkpoint_episode': episode + 1}, step=global_step)

    pygame.quit()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    train(parse_args())

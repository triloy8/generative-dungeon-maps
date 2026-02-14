import argparse
import os

import numpy as np
import pygame
import torch
from tqdm import tqdm

from dqn_agent import DQNAgent
from environment import Environment
from ppo_agent import PPOAgent


class RandomAgent:
    def __init__(self, map_size, value_size, seed=None, mode="uniform"):
        self.map_size = map_size
        self.value_size = value_size
        self.rng = np.random.default_rng(seed)
        self.mode = mode

    def _sample_interior(self):
        x = int(self.rng.integers(1, self.map_size - 1))
        y = int(self.rng.integers(1, self.map_size - 1))
        return x, y

    def act(self, state_tensor):
        if self.mode == "carve":
            map_tensor = state_tensor[0, 0]
            map_np = map_tensor.detach().cpu().numpy()
            solid_positions = np.argwhere(map_np[1:-1, 1:-1] == 1)
            if solid_positions.shape[0] > 0:
                idx = int(self.rng.integers(0, solid_positions.shape[0]))
                y, x = solid_positions[idx]
                return (int(x + 1), int(y + 1)), 0

        # Sample from interior cells to avoid guaranteed no-op border edits.
        x, y = self._sample_interior()
        value = int(self.rng.integers(0, self.value_size))
        return (x, y), value


def resolve_device(arg):
    if arg.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def resolve_dtype(arg):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    if arg not in mapping:
        raise ValueError(f"Unsupported dtype '{arg}'. Choose from {list(mapping.keys())}.")
    return mapping[arg]


def observation_to_tensor(observation, device, dtype):
    stacked = np.stack(
        [
            observation["map"].astype(np.float32),
            observation["heatmap"].astype(np.float32),
        ],
        axis=0,
    )
    return torch.tensor(stacked, dtype=dtype, device=device).unsqueeze(0)


def run_episode(
    env,
    agent,
    device,
    dtype,
    algo='dqn',
    deterministic_ppo=False,
    render=True,
    save_dir=None,
    episode_idx=0,
):
    observation = env.reset()
    first_obs = observation
    state_tensor = observation_to_tensor(observation, device, dtype)
    total_reward = 0.0
    done = False
    last_obs = observation
    for step in tqdm(range(env.max_iterations), desc=f"Episode {episode_idx+1} steps", unit="step", leave=False):
        if render:
            env.render()
            pygame.display.flip()

        if algo == 'dqn':
            action, value = agent.act(state_tensor)
        elif algo == 'random':
            action, value = agent.act(state_tensor)
        else:
            action, value, *_ = agent.act(state_tensor, deterministic=deterministic_ppo)

        next_observation, reward, done = env.step(action, value)
        total_reward += reward
        state_tensor = observation_to_tensor(next_observation, device, dtype)
        last_obs = next_observation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return total_reward, next_observation, True
        if done:
            break
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        env.render()
        final_surface = env.screen.copy()
        original_map = env.map_layout.copy()
        original_colors = [row[:] for row in env.map_colors]
        original_heatmap = env.heatmap.copy()

        env.map_layout = first_obs['map'].copy()
        env.heatmap = first_obs['heatmap'].copy()
        env._update_colors()
        env.render()
        initial_surface = env.screen.copy()

        heatmap_surface = pygame.Surface(env.screen.get_size())
        heatmap_scaled = (last_obs['heatmap'] / max(1.0, env.max_changes)).clip(0.0, 1.0)
        heatmap_pixels = (heatmap_scaled * 255).astype(np.uint8)
        cell_size = 50
        for y in range(env.grid_size):
            for x in range(env.grid_size):
                value = heatmap_pixels[y][x]
                color = (value, value, value)
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                pygame.draw.rect(heatmap_surface, color, rect)

        combined_width = initial_surface.get_width() * 3
        combined_height = initial_surface.get_height()
        combined_surface = pygame.Surface((combined_width, combined_height))
        combined_surface.blit(initial_surface, (0, 0))
        combined_surface.blit(heatmap_surface, (initial_surface.get_width(), 0))
        combined_surface.blit(final_surface, (initial_surface.get_width() * 2, 0))
        pygame.image.save(combined_surface, os.path.join(save_dir, f"episode_{episode_idx:03d}.png"))

        env.map_layout = original_map
        env.map_colors = original_colors
        env.heatmap = original_heatmap
    return total_reward, last_obs, False


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained DQN/PPO agent or a random baseline.")
    parser.add_argument("--algo", choices=['dqn', 'ppo', 'random'], default='dqn', help="Agent type to run.")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model weights (required for DQN/PPO).")
    parser.add_argument("--map-size", type=int, default=7, help="Map dimension (n x n).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of inference episodes to run.")
    parser.add_argument("--target-path", type=int, default=1, help="Target path improvement threshold.")
    parser.add_argument("--render", action="store_true", help="Render the environment with pygame.")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy action selection for PPO inference.")
    parser.add_argument("--save-dir", default=None, help="Directory to save final layout/heatmap PNGs.")
    parser.add_argument("--prob-empty", type=float, default=0.5, help="Initial probability for empty tiles.")
    parser.add_argument("--change-percentage", type=float, default=0.2, help="Fraction of tiles allowed to change.")
    parser.add_argument("--device", default="auto", help="Torch device to use ('auto', 'cpu', 'cuda', etc.).")
    parser.add_argument("--dtype", default="float32", help="Torch dtype (float32, float16, bfloat16, float64).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (used by random baseline).")
    parser.add_argument(
        "--random-mode",
        choices=["uniform", "carve"],
        default="uniform",
        help="Random policy variant: 'uniform' samples random tiles/values, 'carve' only turns solid tiles into walkable tiles.",
    )
    args = parser.parse_args()

    if args.algo in {"dqn", "ppo"} and not args.checkpoint:
        parser.error("--checkpoint is required when --algo is 'dqn' or 'ppo'.")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    pygame.init()
    scrx = args.map_size * 50
    scry = args.map_size * 50
    if args.render:
        screen = pygame.display.set_mode((scrx, scry))
    else:
        screen = pygame.Surface((scrx, scry))

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

    if args.algo == 'dqn':
        agent = DQNAgent(args.map_size, 2, dtype, device)
        agent.load(args.checkpoint)
        agent.epsilon = 0.0
    elif args.algo == 'ppo':
        agent = PPOAgent(args.map_size, dtype, device)
        agent.load(args.checkpoint)
    else:
        agent = RandomAgent(args.map_size, 2, seed=args.seed, mode=args.random_mode)

    for episode in tqdm(range(args.episodes), desc="Inference episodes", unit="episode"):
        total_reward, last_obs, quit_requested = run_episode(
            env,
            agent,
            device,
            dtype,
            algo=args.algo,
            deterministic_ppo=args.deterministic,
            render=args.render,
            save_dir=args.save_dir,
            episode_idx=episode,
        )
        stats = env._get_stats(last_obs["map"])
        frame_info = env.iterations
        print(
            f"Episode {episode + 1}: reward={total_reward:.2f}, "
            f"regions={stats['regions']}, path-length={stats['path-length']}, "
            f"frame={frame_info}"
        )
        if quit_requested:
            break

    pygame.quit()


if __name__ == "__main__":
    main()

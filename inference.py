import argparse
import os

import numpy as np
import pygame
import torch
from tqdm import tqdm

from agent import DQNAgent
from environment import Environment


def observation_to_tensor(observation, device, dtype):
    stacked = np.stack(
        [
            observation["map"].astype(np.float32),
            observation["heatmap"].astype(np.float32),
        ],
        axis=0,
    )
    return torch.tensor(stacked, dtype=dtype, device=device).unsqueeze(0)


def run_episode(env, agent, device, dtype, render=True, save_dir=None, episode_idx=0):
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
        action, value = agent.act(state_tensor)
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
    parser = argparse.ArgumentParser(description="Run inference with a trained DQN agent.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model weights.")
    parser.add_argument("--map-size", type=int, default=7, help="Map dimension (n x n).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of inference episodes to run.")
    parser.add_argument("--target-path", type=int, default=1, help="Target path improvement threshold.")
    parser.add_argument("--render", action="store_true", help="Render the environment with pygame.")
    parser.add_argument("--save-dir", default=None, help="Directory to save final layout/heatmap PNGs.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    pygame.init()
    scrx = args.map_size * 50
    scry = args.map_size * 50
    if args.render:
        screen = pygame.display.set_mode((scrx, scry))
    else:
        screen = pygame.Surface((scrx, scry))

    env = Environment(args.map_size, 2, scrx, scry, screen, args.target_path)
    agent = DQNAgent(args.map_size, 2, dtype, device)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0

    for episode in tqdm(range(args.episodes), desc="Inference episodes", unit="episode"):
        total_reward, last_obs, quit_requested = run_episode(
            env,
            agent,
            device,
            dtype,
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

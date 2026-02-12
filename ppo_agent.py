import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from safetensors.torch import load_file, save_file

from memory import RolloutBuffer


class ActorCriticModel(nn.Module):
    """Shared-backbone actor-critic model for factorized coordinate/tile actions."""

    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )

        self.coord_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )

        self.tile_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(64, 2),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        coord_logits = self.coord_head(feat).flatten(start_dim=1)
        tile_logits = self.tile_head(feat)
        values = self.value_head(feat).squeeze(-1)
        return coord_logits, tile_logits, values


class PPOAgent:
    """PPO agent with on-policy rollout buffer and clipped objective."""

    def __init__(
        self,
        grid_size,
        dtype,
        device,
        gamma=0.95,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=4,
        minibatch_size=64,
        max_grad_norm=0.5,
    ):
        self.grid_size = grid_size
        self.dtype = dtype
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm

        self.model = ActorCriticModel(grid_size).to(device=device, dtype=dtype)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, amsgrad=True)
        self.rollout = RolloutBuffer()

    def _distributions(self, states):
        coord_logits, tile_logits, values = self.model(states)
        coord_dist = Categorical(logits=coord_logits)
        tile_dist = Categorical(logits=tile_logits)
        return coord_dist, tile_dist, values

    def act(self, state, deterministic=False):
        with torch.no_grad():
            coord_dist, tile_dist, values = self._distributions(state)
            if deterministic:
                coord_action = torch.argmax(coord_dist.logits, dim=1)
                tile_action = torch.argmax(tile_dist.logits, dim=1)
            else:
                coord_action = coord_dist.sample()
                tile_action = tile_dist.sample()
            log_prob = coord_dist.log_prob(coord_action) + tile_dist.log_prob(tile_action)

        coord_idx = coord_action.item()
        x = coord_idx % self.grid_size
        y = coord_idx // self.grid_size
        value = int(tile_action.item())
        return [x, y], value, coord_action.unsqueeze(0), tile_action.unsqueeze(0), log_prob.unsqueeze(0), values.unsqueeze(0)

    def remember(self, state, coord_action, tile_action, reward, done, log_prob, value):
        self.rollout.add(
            state=state,
            coord_action=coord_action,
            tile_action=tile_action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value,
        )

    def _compute_gae(self, rewards, dones, values, last_value):
        advantages = torch.zeros_like(rewards, device=self.device, dtype=self.dtype)
        gae = torch.zeros(1, device=self.device, dtype=self.dtype)
        next_value = last_value

        for t in reversed(range(rewards.shape[0])):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def update(self, next_state, next_done):
        if len(self.rollout) == 0:
            return None

        with torch.no_grad():
            _, _, next_value = self.model(next_state)
            if next_done:
                next_value = torch.zeros_like(next_value)

        states, coord_actions, tile_actions, rewards, dones, old_log_probs, values = self.rollout.as_tensors(
            device=self.device,
            dtype=self.dtype,
        )
        values = values.view(-1)
        old_log_probs = old_log_probs.view(-1)
        coord_actions = coord_actions.view(-1)
        tile_actions = tile_actions.view(-1)

        advantages, returns = self._compute_gae(rewards, dones, values, next_value.view(-1))
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False).clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        updates = 0

        batch_size = states.size(0)
        effective_minibatch = min(self.minibatch_size, batch_size)
        idx = torch.arange(batch_size, device=self.device)

        for _ in range(self.ppo_epochs):
            perm = idx[torch.randperm(batch_size)]
            for start in range(0, batch_size, effective_minibatch):
                mb_idx = perm[start:start + effective_minibatch]

                coord_dist, tile_dist, new_values = self._distributions(states[mb_idx])
                new_log_probs = coord_dist.log_prob(coord_actions[mb_idx]) + tile_dist.log_prob(tile_actions[mb_idx])
                entropy = coord_dist.entropy() + tile_dist.entropy()

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                unclipped = ratio * advantages[mb_idx]
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb_idx]
                actor_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (returns[mb_idx] - new_values.view(-1)).pow(2).mean()
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_bonus.item()
                total_loss += loss.item()
                updates += 1

        self.rollout.clear()

        if updates == 0:
            return None

        return {
            "ppo_loss": total_loss / updates,
            "ppo_actor_loss": total_actor_loss / updates,
            "ppo_value_loss": total_value_loss / updates,
            "ppo_entropy": total_entropy / updates,
            "ppo_adv_mean": adv_mean.item(),
            "ppo_adv_std": adv_std.item(),
        }

    def save(self, name):
        save_file(self.model.state_dict(), name)

    def load(self, name):
        state_dict = load_file(name)
        self.model.load_state_dict(state_dict)

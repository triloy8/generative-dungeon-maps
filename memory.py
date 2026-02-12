import random
from collections import namedtuple, deque
import torch

"""A tuple representing a transition in the shape of a namedtuple"""
Transition = namedtuple('Transition',
                        ('state', 'action', 'value', 'reward', 'next_state', 'done'))


class ReplayMemory:
    """This class implements the replay memory used for the agent training.

    It is used as a way to abstract the replay memory using the deque
    object.

    Attributes:
        memory: A deque of Transitions
    """
    def __init__(self, capacity):
        """The init method for the ReplayMemory class, intializing the deque"""
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Size of the current replay memory"""
        return len(self.memory)


class RolloutBuffer:
    """Simple on-policy rollout storage used by PPO-style updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.coord_actions = []
        self.tile_actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, coord_action, tile_action, reward, done, log_prob, value):
        self.states.append(state.detach())
        self.coord_actions.append(coord_action.detach())
        self.tile_actions.append(tile_action.detach())
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())

    def __len__(self):
        return len(self.rewards)

    def as_tensors(self, device, dtype):
        states = torch.cat(self.states, dim=0).to(device=device, dtype=dtype)
        coord_actions = torch.cat(self.coord_actions, dim=0).to(device=device, dtype=torch.long)
        tile_actions = torch.cat(self.tile_actions, dim=0).to(device=device, dtype=torch.long)
        rewards = torch.tensor(self.rewards, device=device, dtype=dtype)
        dones = torch.tensor(self.dones, device=device, dtype=dtype)
        log_probs = torch.cat(self.log_probs, dim=0).to(device=device, dtype=dtype)
        values = torch.cat(self.values, dim=0).to(device=device, dtype=dtype)
        return states, coord_actions, tile_actions, rewards, dones, log_probs, values

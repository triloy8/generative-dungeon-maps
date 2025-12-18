import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
from memory import Transition, ReplayMemory


class DQN(nn.Module):
    """This class implements the DQN model.

    As is it inherited from torch.nn.Module, it implements a forward method for 
    the forward pass describing all the transformations.

    Attributes:
        *_layer*: A layer of the model
        activation: An activation function
    """
    def __init__(self, grid_size, value_size):
        """The init method for the a torch.nn.Module class"""
        super(DQN, self).__init__()
        assert value_size == 2

        # feature extract
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )

        # action head
        self.action_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )
        
        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),          # [B,64,1,1]
            nn.Flatten(1),                    # [B,64]
            nn.Linear(64, value_size)         # [B,2]
        )

    def forward(self, x): 
        """The forward method doing all the transformations"""
        
        feat = self.feature_extractor(x)
        act = self.action_head(feat)
        val = self.value_head(feat)

        return act, val
    
class DQNAgent:
    """This class implements the agent.

    The methods used in this class implement the agent's behavior and its interactions
    with the environment. The agent has a replay memory so that he can learn from previous
    frames, an act method to predict the actiion and value and replay method to train
    the Neural Network for the Deep Q-Learning algorithm.

    Attributes:
        grid_size: An integer representing the width/height of the grid
        value_size: An integer representing the size of the value space (2)
        memory: An instance of a custom ReplayMemory class
        gamma: A float representing the discount factor
        epsilon: A float allowing us to extend an exploration strategy
        epsilon_decay: A float representing the epsilon
        epsilon_min: A float representing the smallest value epsilon can take
        learning_rate: A float used for the learning ratre
        DQNmodel: A torch.nn.Module subclass implementing a Neural Network
        criterion: A torch loss
        optimizer: A torch optimizer
        device: String that tells torch to use either CPU or GPU 

    """
    def __init__(
        self,
        grid_size,
        value_size,
        dtype,
        device,
        memory_capacity=10000,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.01,
        learning_rate=0.00005,
        clip_min=-10.0,
        clip_max=10.0,
        target_update_interval=2000,
    ):
        """The init method for the DQNAgent class"""
        self.device = device
        self.dtype = dtype
        self.grid_size = grid_size
        self.value_size = value_size
        self.memory = ReplayMemory(memory_capacity)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.DQNmodel = DQN(grid_size, value_size).to(device=self.device, dtype=self.dtype)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.DQNmodel.parameters(), lr=self.learning_rate, amsgrad=True)
        self.target_model = DQN(grid_size, value_size).to(device=self.device, dtype=self.dtype)
        self.target_model.load_state_dict(self.DQNmodel.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False
        self.target_update_interval = target_update_interval
        self.train_steps = 0
    
    def remember(self, state, action, value, reward, next_state, done):
        """A method used to update the Replay Memory"""
        self.memory.push(state, action, value, reward, next_state, done)

    def act(self, state):
        """A method used to predict the agents newest action and value based on a current state"""
        if np.random.rand() <= self.epsilon:
            rand_x = random.randrange(self.grid_size)
            rand_y = random.randrange(self.grid_size)
            return [rand_x, rand_y], random.randrange(self.value_size)
        with torch.no_grad():
            action, value = self.DQNmodel(state)
        if (torch.max(action[0])==0):
            rand_x = random.randrange(self.grid_size)
            rand_y = random.randrange(self.grid_size)
            return [rand_x, rand_y], random.randrange(self.value_size)
        else:
            grid = action[0, 0]  # drop the channel axis -> grid_size x grid_size
            max_coords = torch.nonzero(grid == grid.max(), as_tuple=False)
            row, col = max_coords[0].tolist()
            action_item = [col, row]
            value_item = torch.argmax(value[0]).item()
        
        return action_item, value_item

    def replay(self, batch_size):
        """We're using a batch of from the Replay Memory to train the DQNAgent"""
        transitions = self.memory.sample(batch_size)
        minibatch = Transition(*zip(*transitions))

        state_batch = torch.cat(minibatch.state)
        action_batch = torch.cat(minibatch.action)
        value_batch = torch.cat(minibatch.value)
        reward_batch = torch.cat(minibatch.reward)
        next_state_batch = torch.cat(minibatch.next_state)
        done_batch = torch.cat(minibatch.done)

        with torch.no_grad():
            next_online_act, next_online_val = self.DQNmodel(next_state_batch.to(self.device))
            next_target_act, next_target_val = self.target_model(next_state_batch.to(self.device))

        # current states must keep grads for the update
        curr_act, curr_val = self.DQNmodel(state_batch.to(self.device))

        # clone/detach so we can write target values without affecting the graph
        target_act = curr_act.detach().clone()
        target_val = curr_val.detach().clone()

        batch_indices = torch.arange(batch_size, device=self.device)
        best_action_flat = next_online_act.view(batch_size, -1).argmax(dim=1)
        best_action_rows = (best_action_flat // self.grid_size).long()
        best_action_cols = (best_action_flat % self.grid_size).long()
        next_q_targets = next_target_act[batch_indices, 0, best_action_rows, best_action_cols]
        best_value_indices = torch.argmax(next_online_val, dim=1)
        next_value_targets = next_target_val[batch_indices, best_value_indices]
        not_done = 1 - done_batch.to(self.device).float()
        rewards = reward_batch.to(self.device)

        td_action = rewards + not_done * self.gamma * next_q_targets
        td_value = rewards + not_done * self.gamma * next_value_targets

        td_action = td_action.clamp_(min=self.clip_min, max=self.clip_max)
        td_value = td_value.clamp_(min=self.clip_min, max=self.clip_max)

        for i in range(batch_size):
            target_act[i, 0, action_batch[i][1].item(), action_batch[i][0].item()] = td_action[i].item()
            target_val[i, value_batch[i].item()] = td_value[i].item()

        arange_idx = torch.arange(batch_size, device=self.device)
        row_idx = action_batch[:, 1].long()
        col_idx = action_batch[:, 0].long()
        pred_q = curr_act[arange_idx, 0, row_idx, col_idx]
        target_q = target_act[arange_idx, 0, row_idx, col_idx]
        action_loss = self.criterion(pred_q, target_q)

        loss = action_loss + self.criterion(curr_val, target_val)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.DQNmodel.state_dict())
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        debug_stats = {
            "curr_act_max": curr_act.detach().abs().max().item(),
            "next_act_max": next_online_act.detach().abs().max().item(),
            "target_act_max": target_act.abs().max().item(),
            "reward_max": reward_batch.abs().max().item(),
            "curr_val_max": curr_val.detach().abs().max().item(),
            "target_val_max": target_val.abs().max().item(),
        }
        
        return loss.item(), debug_stats
    
    def save(self, name):
        """Saving the model weights"""
        save_file(self.DQNmodel.state_dict(), name)
    
    def load(self, name):
        """Loading the model weights for inference"""
        state_dict = load_file(name)
        self.DQNmodel.load_state_dict(state_dict)

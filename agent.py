import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from memory import Transition, ReplayMemory


class DQN(nn.Module):
    """This class implements the DQN model.

    As is it inherited from torch.nn.Module, it implements a forward method for 
    the forward pass describing all the transformations.

    Attributes:
        *_layer*: A layer of the model
        activation: An activation function
    """
    def __init__(self, state_size, action_size, value_size):
        """The init method for the a torch.nn.Module class"""
        super(DQN, self).__init__()
        self.kernel_size = 3
        self.stride = 1 
        self.same_padding = ((state_size-1)*self.stride-state_size+self.kernel_size)//2
        self.action_layer1 = nn.Conv2d(1, 32, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer2 = nn.Conv2d(32, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer3 = nn.Conv2d(64, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer4 = nn.Conv2d(64, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer5 = nn.Conv2d(64, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer6 = nn.Conv2d(64, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer7 = nn.Conv2d(64, 64, self.kernel_size, self.stride, padding=self.same_padding)
        self.action_layer8 = nn.Conv2d(64, 1, self.kernel_size, self.stride, padding=self.same_padding)

        self.value_layer1 = nn.Conv2d(1, 64, kernel_size=3, stride=2)
        self.value_layer2 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.value_layer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.linear_input_size = (action_size - 3)//4 
        self.value_layer4 = nn.Linear(64*(self.linear_input_size)**2, value_size)

        self.activation = nn.ReLU()

    def forward(self, x): 
        """The forward method doing all the forward transformations"""
        # Action transformations
        x = self.activation(self.action_layer1(x))
        x = self.activation(self.action_layer2(x))
        x = self.activation(self.action_layer3(x))
        x = self.activation(self.action_layer4(x))
        x = self.activation(self.action_layer5(x))
        x = self.activation(self.action_layer6(x))
        x = self.activation(self.action_layer7(x))
        x = self.activation(self.action_layer8(x))
        act = x

        # Value transformations
        x = self.activation(self.value_layer1(x))
        x = self.activation(self.value_layer2(x))
        x = self.activation(self.value_layer3(x))
        val = torch.flatten(x, start_dim=1)
        val = self.value_layer4(val)

        return act, val
    
class DQNAgent:
    """This class implements the agent.

    The methods used in this class implement the agent's behavior and its interactions
    with the environment. The agent has a replay memory so that he can learn from previous
    frames, an act method to predict the actiion and value and replay method to train
    the Neural Network for the Deep Q-Learning algorithm.

    Attributes:
        state_size: An integer represting the size of the state space (n*n)
        action_size: An integer represting the size of the action space (n*n)
        value_size: An integer represting the size of the value space (2)
        memory: An instance of a custom ReplaMemory class
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
    def __init__(self, state_size, action_size, value_size, model, device):
        """The init method for the DQNAgent class"""
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = value_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.DQNmodel = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.DQNmodel.parameters(), lr=self.learning_rate, amsgrad=True)
        self.device = device
    
    def remember(self, state, action, value, reward, next_state, done):
        """A method used to update the Replay Memory"""
        self.memory.push(state, action, reward, value, next_state, done)

    def act(self, state):
        """A method used to predict the agents newest action and value based on a current state"""
        if np.random.rand() <= self.epsilon:
            return random.sample(range(self.state_size), 2), random.randrange(self.value_size)
        with torch.no_grad():
            action, value = self.DQNmodel(state.to(self.device))
        if (torch.max(action[0])==0):
            return random.sample(range(self.state_size), 2), random.randrange(self.value_size)
        else:
            action_item = [(action[0]==torch.max(action[0])).nonzero().squeeze()[0].item(), \
                (action[0]==torch.max(action[0])).nonzero().squeeze()[1].item()]
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
            predicted_action_batch, predicted_value_batch = self.DQNmodel(next_state_batch.to(self.device))
            predicted_action_batch_target, predicted_value_batch_target = self.DQNmodel(state_batch.to(self.device))
   
        for i, done in enumerate(done_batch):
            if done.item():
                target_action = reward_batch[i].item()
                target_value = reward_batch[i].item()
            if not done.item():
                target_action = (reward_batch[i].item() + self.gamma * torch.max(predicted_action_batch[i]).item())
                target_value = (reward_batch[i].item() + self.gamma * torch.max(predicted_value_batch[i]).item())
            predicted_action_batch_target[i][0][action_batch[i][0].item()][action_batch[i][1].item()], \
            predicted_value_batch_target[i][value_batch[i].item()] = \
            target_action, target_value

        diff_predicted_action_batch = predicted_action_batch.clone().detach().requires_grad_(True)
        diff_predicted_action_batch_target = predicted_action_batch_target.clone().detach().requires_grad_(True)
        
        diff_predicted_value_batch = predicted_value_batch.clone().detach().requires_grad_(True)
        diff_predicted_value_batch_target = predicted_value_batch_target.clone().detach().requires_grad_(True)

        loss = self.criterion(diff_predicted_action_batch, diff_predicted_action_batch_target) + \
            self.criterion(diff_predicted_value_batch, diff_predicted_value_batch_target)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, name):
        """Saving the model weights"""
        torch.save(self.DQNmodel.state_dict(), name)
    
    def load(self, name):
        """Loading the model weights for inference"""
        self.DQNmodel.load_state_dict(torch.load(name))
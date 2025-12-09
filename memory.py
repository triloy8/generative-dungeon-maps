import random
from collections import namedtuple, deque

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
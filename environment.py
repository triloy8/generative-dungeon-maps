import numpy as np
import random
import pygame
from helper import get_range_reward, calc_longest_path, calc_num_regions

class Environment:
    """This class implements the environment.

    The methods used in this class are inspired by the way the environment in the
    Gymnasium package implements them: a render method for the graphics, a reset
    method to initialize attributes

    Attributes:
        state_size: An integer represting the size of the state space (n*n)
        action_size: An integer represting the size of the action space (n*n)
        value_size: An integer represting the size of the value space (2)
        num_initial_walkable_tiles: An integer of the number of initial walkable tiles
        target_path: An integer ofthe size of the longest shortest path
        *_color: A tuple of RGB colors used for the rendering
        rewards: A dictionary of the reward types
        scrx/ scry: PyGame screen sizes
        screen: The PyGame Screen
    """
    def __init__(self, state_size, action_size, value_size, scrx, scry, screen, initial_walkable_tiles, target_path):
        """The init method for the Environment class"""
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = value_size
        self.num_initial_walkable_tiles = initial_walkable_tiles
        self.target_path = target_path
        self.walkable_tile_color = (51,51,51) 
        self.brick_tile_color = (220, 85, 57)
        self.start_tile_color = (18, 217, 0)
        self.finish_tile_color = (247, 13, 26)
        self.treasure_tile_color = (255, 215, 0)
        self.rewards = {
            "regions": 5,
            "path-length": 1
        }
        self.scrx = scrx
        self.scry = scry
        self.screen = screen


    
    def render(self):
        """This method is solely responsible for the rendering of the graphics using PyGame"""
        for i in range(0, self.scrx, 50):
            for j in range(0, self.scry, 50):
                pygame.draw.rect(self.screen,(255,255,255),(j,i,j+50,i+50),0)
                pygame.draw.rect(self.screen,self.map_colors[i//50][j//50],(j+2,i+2,j+48,i+48),0)
    
    def reset(self):
        """The reset method is used to reset some the method attributes during training for each new episode"""
        # Initializing the map layout, colors and walkable tiles
        self.initial_map_layout = np.ones((self.state_size, self.state_size))
        self.initial_map_colors = [[self.brick_tile_color for i in range(self.state_size) ] for j in range(self.state_size)]
        self.all_tiles = {(i, j) for i in range(self.state_size) for j in range(self.state_size)}
        self.initial_walkable_tiles = random.sample(list(self.all_tiles), self.num_initial_walkable_tiles)
        for i in range(self.num_initial_walkable_tiles):
            x = self.initial_walkable_tiles[i][0]
            y = self.initial_walkable_tiles[i][1]
            self.initial_map_layout[x][y] = 0
            self.initial_map_colors[x][y] = self.walkable_tile_color

        # Initializing moving variables
        self.map_layout = self.initial_map_layout
        self.walkable_tiles = self.initial_walkable_tiles
        self.map_colors = self.initial_map_colors
        self.initial_path_length = calc_longest_path(self.initial_map_layout, self.walkable_tiles)
        return self.map_layout
    
    def step(self, action, value):   
        """
        The step method is used to transition from a state to another for a timestep all the while calculating 
        the reward and the done status
        """      
        old_map_layout = self.map_layout
        old_path_length = calc_longest_path(old_map_layout, self.walkable_tiles)
        old_num_regions = calc_num_regions(old_map_layout, self.walkable_tiles)
        
        self.map_layout[action[0]][action[1]] = value
        self.walkable_tiles = [(i, j) for i in range(self.state_size)
                                  for j in range(self.state_size)
                                  if self.map_layout[i][j] == 0]

        new_map_layout = self.map_layout
        new_path_length = calc_longest_path(new_map_layout, self.walkable_tiles)
        new_num_regions = calc_num_regions(new_map_layout, self.walkable_tiles)
        for i in range(len(new_map_layout)):
            for j in range(len(new_map_layout)):
                if new_map_layout[i][j] == 0:
                    self.map_colors[i][j] = self.walkable_tile_color
                else:
                    self.map_colors[i][j] = self.brick_tile_color

        done = int(new_num_regions == 1 and new_path_length - self.initial_path_length >= self.target_path)
        
        path_length_range_reward = get_range_reward(new_path_length, old_path_length, np.inf, np.inf)
        num_regions_range_reward = get_range_reward(new_num_regions, old_num_regions, 1, 1)

        reward = num_regions_range_reward * self.rewards["regions"] + path_length_range_reward * self.rewards["path-length"]
        
        return new_map_layout, reward, done
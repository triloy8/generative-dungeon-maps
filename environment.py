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
        grid_size: An integer representing the width/height of the square grid
        value_size: An integer represting the size of the value space (2)
        target_path: An integer ofthe size of the longest shortest path
        *_color: A tuple of RGB colors used for the rendering
        rewards: A dictionary of the reward types
        scrx/ scry: PyGame screen sizes
        screen: The PyGame Screen
    """
    def __init__(self, grid_size, value_size, scrx, scry, screen, target_path, prob_empty=0.5, change_percentage=0.2):
        """The init method for the Environment class"""
        self.grid_size = grid_size
        self.value_size = value_size
        self.empty_value = 0
        self.solid_value = 1
        self.prob_empty = min(1.0, max(0.0, prob_empty))
        self.prob_solid = 1 - self.prob_empty
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
        self.change_percentage = min(1.0, max(0.0, change_percentage))
        total_tiles = max(1, self.grid_size * self.grid_size)
        self.max_changes = max(1, int(self.change_percentage * total_tiles))
        self.max_iterations = self.max_changes * total_tiles
        self.changes = 0
        self.iterations = 0
        self.budget_exhausted = False
        self.heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

    def render(self):
        """This method is only responsible for the rendering of the graphics using PyGame"""
        for i in range(0, self.scrx, 50):
            for j in range(0, self.scry, 50):
                pygame.draw.rect(self.screen,(255,255,255),(j,i,j+50,i+50),0)
                pygame.draw.rect(self.screen,self.map_colors[i//50][j//50],(j+2,i+2,j+48,i+48),0)
    
    def reset(self):
        """The reset method is used to reset some the method attributes during training for each new episode"""
        # Initializing the map layout, colors and walkable tiles
        self.initial_map_layout, self.initial_map_colors = self._generate_initial_layout()
        self.map_layout = self.initial_map_layout.copy()
        self.map_colors = [row[:] for row in self.initial_map_colors]
        self.walkable_tiles = self._get_walkable_tiles(self.map_layout)
        self.start_stats = self._get_stats(self.map_layout)
        self.initial_path_length = self.start_stats["path-length"]
        self.changes = 0
        self.iterations = 0
        self.budget_exhausted = False
        self.heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        return self._get_observation()
    
    def step(self, action, value):   
        """
        The step method is used to transition from a state to another for a timestep all the while calculating 
        the reward and the done status
        """      
        self.iterations += 1
        old_map_layout = self.map_layout.copy()
        old_stats = self._get_stats(old_map_layout)

        x, y = action
        previous_value = self.map_layout[y][x]
        self.map_layout[y][x] = value
        if previous_value != value:
            self.changes += 1
        self.walkable_tiles = self._get_walkable_tiles(self.map_layout)
        self._update_colors()

        new_map_layout = self.map_layout
        new_stats = self._get_stats(new_map_layout)

        reward = self._compute_reward(new_stats, old_stats)
        budget_exhausted = self.changes >= self.max_changes or self.iterations >= self.max_iterations
        self.budget_exhausted = budget_exhausted
        done = int(self._is_episode_over(new_stats) or budget_exhausted)

        if previous_value != value:
            self.heatmap[y][x] = min(self.heatmap[y][x] + 1.0, self.max_changes)

        observation = self._get_observation()

        return observation, reward, done

    def _generate_initial_layout(self):
        layout = np.full((self.grid_size, self.grid_size), self.solid_value, dtype=np.int8)
        colors = [[self.brick_tile_color for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_border(x, y):
                    continue
                tile_value = self.empty_value if random.random() < self.prob_empty else self.solid_value
                layout[y][x] = tile_value
                colors[y][x] = self.walkable_tile_color if tile_value == self.empty_value else self.brick_tile_color
        return layout, colors

    def _is_border(self, x, y):
        return x == 0 or y == 0 or x == self.grid_size - 1 or y == self.grid_size - 1

    def _get_walkable_tiles(self, layout=None):
        target_layout = self.map_layout if layout is None else layout
        return [
            (x, y)
            for y in range(self.grid_size)
            for x in range(self.grid_size)
            if target_layout[y][x] == self.empty_value
        ]

    def _get_stats(self, layout=None):
        target_layout = self.map_layout if layout is None else layout
        walkable = self._get_walkable_tiles(target_layout)
        return {
            "regions": calc_num_regions(target_layout, walkable),
            "path-length": calc_longest_path(target_layout, walkable),
        }

    def _compute_reward(self, new_stats, old_stats):
        path_length_range_reward = get_range_reward(new_stats["path-length"], old_stats["path-length"], np.inf, np.inf)
        num_regions_range_reward = get_range_reward(new_stats["regions"], old_stats["regions"], 1, 1)
        return num_regions_range_reward * self.rewards["regions"] + path_length_range_reward * self.rewards["path-length"]

    def _is_episode_over(self, new_stats):
        if self.start_stats is None:
            return False
        path_improvement = new_stats["path-length"] - self.start_stats["path-length"]
        return new_stats["regions"] == 1 and path_improvement >= self.target_path

    def _update_colors(self):
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.map_layout[y][x] == self.empty_value:
                    self.map_colors[y][x] = self.walkable_tile_color
                else:
                    self.map_colors[y][x] = self.brick_tile_color

    def _get_observation(self):
        return {
            "map": self.map_layout.copy(),
            "heatmap": self.heatmap.copy(),
        }

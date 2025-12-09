import numpy as np

"""
Function that runs dikjstra algorithm and return the map

Parameters:
    x (int): the starting x position for dikjstra algorithm
    y (int): the starting y position for dikjstra algorithm
    map (any[][]): the current map being tested

Returns:
    int[][]: returns the dikjstra map after running the dijkstra algorithm
"""
def run_dikjstra(x, y, map_layout):
    dikjstra_map = np.full((len(map_layout), len(map_layout[0])),-1)
    visited_map = np.zeros((len(map_layout), len(map_layout[0])))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx,cy,cd) = queue.pop(0)
        if map_layout[cy][cx] not in [0] or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map_layout[0]) or ny >= len(map_layout):
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map

"""
Calculate the approximate longest shortest path (i.e. all pairs shortest path) on the map.

Parameters:
    map_layout (any[][]): the current map being tested
    walkable_tiles (Dict((int,int))): walkable tiles of the map

Returns:
    int: the longest path in tiles in the current map
"""
def calc_longest_path(map_layout, walkable_tiles):
    final_visited_map = np.zeros((len(map_layout), len(map_layout[0])))
    final_value = 0
    for (x,y) in walkable_tiles:
        if final_visited_map[y][x] > 0:
            continue
        dikjstra_map, visited_map = run_dikjstra(x, y, map_layout)
        final_visited_map += visited_map
        (my,mx) = np.unravel_index(np.argmax(dikjstra_map, axis=None), dikjstra_map.shape)
        dikjstra_map, _ = run_dikjstra(mx, my, map_layout)
        max_value = np.max(dikjstra_map)
        if max_value > final_value:
            final_value = max_value
    return final_value

"""
Function that runs flood fill algorithm on the current color map

Parameters:
    x (int): the starting x position of the flood fill algorithm
    y (int): the starting y position of the flood fill algorithm
    color_map (int[][]): the color map that is being colored
    map_layout (any[][]): the current tile map to check
    color_index (int): the color used to color in the color map

Returns:
    int: the number of tiles that has been colored
"""
def flood_fill(x, y, color_map, map_layout, color_index):
    num_tiles = 0
    queue = [(x, y)]
    while len(queue) > 0:
        (cx, cy) = queue.pop(0)
        if color_map[cy][cx] != -1 or map_layout[cy][cx] not in [0]:
            continue
        num_tiles += 1
        color_map[cy][cx] = color_index
        for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx,ny=cx+dx,cy+dy
            if nx < 0 or ny < 0 or nx >= len(map_layout[0]) or ny >= len(map_layout):
                continue
            queue.append((nx, ny))
    return num_tiles

"""
Calculates the number of regions in the current map with passable_values

Parameters:
    map_layout (any[][]): the current map being tested
    walkable_tiles (Dict((int,int))): walkable tiles of the map

Returns:
    int: number of regions in the map
"""
def calc_num_regions(map_layout, walkable_tiles):
    region_index=0
    color_map = np.full((len(map_layout), len(map_layout[0])), -1)
    for (x,y) in walkable_tiles:
        num_tiles = flood_fill(x, y, color_map, map_layout, region_index + 1)
        if num_tiles > 0:
            region_index += 1
        else:
            continue
    return region_index

"""
A method to help calculate the reward value based on the change around optimal region

Parameters:
    new_value (float): the new value to be checked
    old_value (float): the old value to be checked
    low (float): low bound for the optimal region
    high (float): high bound for the optimal region

Returns:
    float: the reward value for the change between new_value and old_value
"""
def get_range_reward(new_value, old_value, low, high):
    if new_value >= low and new_value <= high and old_value >= low and old_value <= high:
        return 0
    if old_value <= high and new_value <= high:
        return min(new_value, low) - min(old_value, low)
    if old_value >= low and new_value >= low:
        return max(old_value, high) - max(new_value, high)
    if new_value > high and old_value < low:
        return high - new_value + old_value - low
    if new_value < low and old_value > high:
        return high - old_value + new_value - low
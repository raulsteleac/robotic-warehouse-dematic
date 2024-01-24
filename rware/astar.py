import heapq

def heuristic(a, b):
    """
    Heuristic function for A* algorithm (Manhattan distance).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    """
    A* algorithm for finding the optimal path in a grid.

    Parameters:
    - grid (2D list): The grid where 0 represents open space and 1 represents obstacles.
    - start (tuple): Starting point (x, y).
    - goal (tuple): Goal point (x, y).

    Returns:
    - path (list): Optimal path from start to goal.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break
        for neighbor in neighbors(grid, current_node):
            new_cost = cost_so_far[current_node] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node
    path = reconstruct_path(came_from, start, goal)
    return path[1:]

def neighbors(grid, node):
    """
    Get neighboring nodes of a given node in the grid.
    """
    x, y = node
    potential_neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [(nx, ny) for nx, ny in potential_neighbors if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] == 0]

def reconstruct_path(came_from, start, goal):
    """
    Reconstruct the optimal path from the start to the goal.
    """
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
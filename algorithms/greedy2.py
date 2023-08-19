import math
from functions import *
def manhattan_distance(node1, node2):
    """Calculate the Manhattan distance between two nodes."""
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])


def compute_score(start, end, travel_times, packages_sent, recoveries, penalties):
    """Compute the final score using the provided formula."""
    
    # Total distance traveled
    D = sum(travel_times)
    
    # Distance between the start and end points
    d = manhattan_distance(start, end)
    
    # Total packages sent
    P = sum(packages_sent)
    
    # Total recovery minutes
    T = sum(recoveries)
    
    # Total weight penalty
    W = sum(penalties)
    
    # Compute the score using the formula
    SF = 20 ** (math.log(0.1 * d * (D + 0.01) / d) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    return round(SF)

def food_required(travel_time):
    """Calculate the number of food packages required given a travel time."""
    # If travel time is a multiple of 10, divide by 10
    # Otherwise, divide by 10 and round up
    out = travel_time// 10
    rem = travel_time % 10
    if rem >= 5:
        out = out + 1
    return out

def weight_penalty(travel_time, packages):
    """Calculate the weight penalty for carrying food packages over a given travel time."""
    penalty = 0
    remaining_time = travel_time
    
    while remaining_time > 0:
        penalty += packages
        remaining_time -= 10
        if remaining_time > 0:  # If there's still travel time left, consume a package
            packages -= 1

    return penalty

def recovery_mins(travel_time):
    t = math.ceil(travel_time/10)
    rem = travel_time % 10
    rec = 0
    if rem >= 5:
        rec = t*10-travel_time
    return rec
        


def greedy_path_with_packages(nodes, start, end, package_function):
    """Greedy algorithm to find the path based on shortest Manhattan distance and then compute packages."""
    
    # Step 1: Initialization
    remaining_nodes = nodes.copy()
    current_node = start
    path = []
    
    # Step 2: Greedy Selection
    while remaining_nodes:
        # Calculate Manhattan distance to all unvisited nodes
        distances = [manhattan_distance(current_node, node) for node in remaining_nodes]
        
        # Find the node with the shortest distance
        next_node = remaining_nodes[distances.index(min(distances))]
        path.append(next_node)
        remaining_nodes.remove(next_node)
        current_node = next_node
    
    # Add the end node to the path
    path.append(end)
    
    # Step 3: Package Calculation
    packages_needed = []
    current_node = start
    for next_node in path:
        travel_time = manhattan_distance(current_node, next_node)
        packages = package_function(travel_time)
        packages_needed.append(packages)
        current_node = next_node
    
    return path, packages_needed


def greedy_path_with_packages_and_modified_score(nodes, start, end, package_function):
    """Greedy algorithm to find the path based on shortest Manhattan distance, compute packages, and then score."""
    
    # Step 1: Initialization
    remaining_nodes = nodes.copy()
    current_node = start
    path = []
    
    # Step 2: Greedy Selection
    while remaining_nodes:
        # Calculate Manhattan distance to all unvisited nodes
        distances = [manhattan_distance(current_node, node) for node in remaining_nodes]
        
        # Find the node with the shortest distance
        next_node = remaining_nodes[distances.index(min(distances))]
        path.append(next_node)
        remaining_nodes.remove(next_node)
        current_node = next_node
    
    # Add the end node to the path
    path.append(end)
    
    # Step 3: Package Calculation
    travel_times = []
    packages_needed = []
    current_node = start
    for next_node in path:
        travel_time = manhattan_distance(current_node, next_node)
        travel_times.append(travel_time)
        packages = package_function(travel_time)
        packages_needed.append(packages)
        current_node = next_node
    
    # Step 4: Score Calculation using the modified function
    travel_times, packages_sent, recoveries, penalties = compute_journey_info(start, path)
    score = modified_compute_score(start, end, travel_times, packages_sent, recoveries, penalties)
    
    return path, packages_needed, score

test_nodes = [(7, 6), (3, 5), (9, 8)]
test_start = (0, 0)
test_end = (10, 10)
# Testing the algorithm
path_greedy, packages_greedy = greedy_path_with_packages(test_nodes, test_start, test_end, food_required)
path_greedy, packages_greedy



print(compute_score(test_start,test_end,))

print(path_greedy)
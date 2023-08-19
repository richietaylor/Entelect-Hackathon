
import math

def manhattan_distance(node1, node2):
    """Calculate the Manhattan distance between two nodes."""
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def required_packages(travel_time):
    """Calculate the number of food packages needed given travel time."""
    return (travel_time + 9) // 10

def weight_penalty(packages, travel_time):
    """Calculate the weight penalty given the number of packages and travel time."""
    penalty = 0
    while travel_time > 0:
        penalty += packages
        packages -= 1
        travel_time -= 10
    return penalty

def recovery_time(packages):
    """Calculate the recovery time given the number of packages."""
    return 10 * packages

def nearest_neighbor_path(locations):
    """Determine the path using the nearest neighbor heuristic."""
    start_node = (0, 0)
    path = [start_node]
    unvisited_nodes = set(locations)
    
    while unvisited_nodes:
        current_node = path[-1]
        # Find the nearest unvisited node
        nearest_node = min(unvisited_nodes, key=lambda x: manhattan_distance(current_node, x))
        path.append(nearest_node)
        unvisited_nodes.remove(nearest_node)
    
    return path

def generate_submission(path):
    """Generate the submission list based on the path."""
    submission = []
    current_node = path[0]
    for i in range(1, len(path)):
        # Calculate travel details
        travel_time = manhattan_distance(current_node, path[i])
        packages = required_packages(travel_time)
        penalty = weight_penalty(packages, travel_time)
        recovery = recovery_time(packages)
        
        # Append to submission
        submission.append([packages, [path[i]]])
        print(packages +100, path[i])
        
        # Update current node
        current_node = path[i]
    
    return submission

def calculate_score(submission, locations):
    """Calculate the score based on the submission and locations."""
    # Initial values
    D = 0  # Total distance travelled
    P = 0  # Total packages sent
    T = 0  # Total recovery minutes
    W = 0  # Total weight penalty
    
    current_node = (0, 0)
    for entry in submission:
        packages, destinations = entry
        P += packages
        T += recovery_time(packages)
        for destination in destinations:
            travel_time = manhattan_distance(current_node, destination)
            D += travel_time
            W += weight_penalty(packages, travel_time)
            # Adjust packages for consumption
            packages -= required_packages(travel_time)
            current_node = destination
    
    d = manhattan_distance((0, 0), current_node)
    
    # Calculate final score using the given formula
    SF = 20 * (math.log(0.1 * d * (D + 0.01) / d) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    
    return round(SF)

# Test the code with the provided example
locations_example = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
path_example = nearest_neighbor_path(locations_example)
submission_example = generate_submission(path_example)
score_example = calculate_score(submission_example, locations_example)
print(score_example)

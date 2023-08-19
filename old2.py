
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
        nearest_node = min(unvisited_nodes, key=lambda x: manhattan_distance(current_node, x))
        path.append(nearest_node)
        unvisited_nodes.remove(nearest_node)
    
    return path

def generate_submission(path):
    """Generate the submission list based on the path."""
    submission = []
    current_node = path[0]
    for i in range(1, len(path)):
        travel_time = manhattan_distance(current_node, path[i])
        packages = required_packages(travel_time)
        penalty = weight_penalty(packages, travel_time)
        recovery = recovery_time(packages)
        
        submission.append([packages, [path[i]]])
        current_node = path[i]
    
    return submission

def calculate_score(submission, locations):
    """Calculate the score based on the submission and locations."""
    D = 0  
    P = 0  
    T = 0  
    W = 0  
    
    current_node = (0, 0)
    for entry in submission:
        packages, destinations = entry
        P += packages
        T += recovery_time(packages)
        for destination in destinations:
            travel_time = manhattan_distance(current_node, destination)
            D += travel_time
            W += weight_penalty(packages, travel_time)
            packages -= required_packages(travel_time)
            current_node = destination
    
    d = manhattan_distance((0, 0), current_node)
    SF = 20 * (math.log(0.1 * d * (D + 0.01) / d) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    
    return round(SF)

# Example usage:
if __name__ == "__main__":
    locations_example = [(42, 43), (12, 45), (35, 49), (31, 32), (50, 50)]
    path_example = nearest_neighbor_path(locations_example)
    print("Path:", path_example)
    submission_example = generate_submission(path_example)
    print("Submission:", submission_example)
    score_example = calculate_score(submission_example, locations_example)
    print("Score:", score_example)



def optimize_with_buffer(path):
    """Optimize the number of packages sent with a buffer."""
    optimized_submission = []
    current_node = path[0]
    buffer = 10
    for i in range(1, len(path)):
        travel_time = manhattan_distance(current_node, path[i])
        exact_packages = required_packages(travel_time)
        if i < len(path) - 1:
            next_travel_time = manhattan_distance(path[i], path[i+1])
            buffer = required_packages(next_travel_time) - 1
        
        total_packages = exact_packages + buffer
        optimized_submission.append([total_packages, [path[i]]])
        current_node = path[i]
    
    return optimized_submission

# Updated example usage to write the results to a txt file
if __name__ == "__main__":
    locations_example = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
    path_example = nearest_neighbor_path(locations_example)
    
    # optimized_submission_example = optimize_packages(path_example)
    # optimized_score_example = calculate_score(optimized_submission_example, locations_example)
    
    optimized_with_buffer_submission = optimize_with_buffer(path_example)
    print(optimize_with_buffer(path_example))
    optimized_with_buffer_score = calculate_score(optimized_with_buffer_submission, locations_example)
    print(calculate_score(optimized_with_buffer_submission, locations_example))
    
    with open("expedition_results.txt", "w") as file:
        file.write("Path:\n" + str(path_example) + "\n")
        # file.write("Optimized Submission (Exact Needs):\n" + str(optimized_submission_example) + "\n")
        # file.write("Score (Exact Needs): " + str(optimized_score_example) + "\n")
        file.write("Optimized Submission (With Buffer):\n" + str(optimized_with_buffer_submission) + "\n")
        file.write("Score (With Buffer): " + str(optimized_with_buffer_score))


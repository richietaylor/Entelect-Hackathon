
import math
from ant import *

def manhattan_distance(point1, point2):
    """Calculate the Manhattan distance between two points."""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def find_greedy_path(start, nodes):
    """Find a path using a greedy approach, always moving to the nearest unvisited node."""
    path = [start]
    while nodes:
        current = path[-1]
        # Find the nearest unvisited node
        nearest_node = min(nodes, key=lambda x: manhattan_distance(current, x))
        path.append(nearest_node)
        nodes.remove(nearest_node)
    return path

def calculate_packages_required(distance):
    """Calculate the number of food packages required based on the distance."""
    # If travel time is a multiple of 10, divide by 10
    # Otherwise, divide by 10 and round up
    out = distance// 10
    rem = distance % 10
    if rem >= 5:
        out = out + 1
    return out

def calculate_weight_penalty(distance, packages):
    """Calculate the weight penalty for the journey based on the number of packages."""
    penalty = 0
    while distance > 0 and packages > 0:
        penalty += packages
        distance -= 10
        packages -= 1
    return penalty

def journey_details(path):
    """Calculate the packages required, weight penalty, and recovery time for a given path."""
    total_distance = 0
    total_packages = 0
    total_penalty = 0
    total_recovery = 0
    details = []
    for i in range(len(path) - 1):
        start, end = path[i], path[i+1]
        distance = manhattan_distance(start, end)
        packages = calculate_packages_required(distance)
        penalty = calculate_weight_penalty(distance, packages)
        total_distance += distance
        total_packages += packages
        total_penalty += penalty
        total_recovery += packages * 10
        details.append({
            'start': start,
            'end': end,
            'distance': distance,
            'packages': packages,
            'penalty': penalty,
            'recovery': packages * 10
        })
    return details, total_distance, total_packages, total_penalty, total_recovery

def calculate_score(D, d, P, T, W):
    """Calculate the final score using the provided formula."""
    score = 20 * (math.log(0.1 * d * (D + 0.01 / d)) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    return round(score)

# Example usage
if __name__ == "__main__":
    # Given nexus locations
    nexus_locations = [(90, 184), (148, 66), (175, 118), (123, 71), (153, 53), (71, 65), (83, 66), (47, 42), (87, 26), (193, 45), (200, 200)]
    start = (0, 0)
    # end = (10, 10)
    # Find the path using the greedy approach
    greedy_path = find_greedy_path(start, nexus_locations)
    print("Greedy Path:", greedy_path)
    journey_data, total_distance, total_packages, total_penalty, total_recovery = journey_details(greedy_path)

    # aco = AntColonyOptimization(start, nexus_locations) #don't forget end
    # print("Ant Path:", )
    # journey_data, total_distance, total_packages, total_penalty, total_recovery = journey_details(greedy_path)

    print("Journey Details:", journey_data)
    d = 100  # Distance between start and end points
    # Calculate the score
    score = calculate_score(total_distance, d, total_packages, total_recovery, total_penalty)
    print("Score:", score)


# Generate the output in the desired format
output_content = ["["]
for data in journey_data:
    line = f"	[{data['packages']}, [{data['end']}]],"
    output_content.append(line)
output_content.append("]")

# Save the output to a .txt file
output_path = "output.txt"
with open(output_path, "w") as file:
    for line in output_content:
        file.write(line + "\n")

print("Output saved to output.txt")

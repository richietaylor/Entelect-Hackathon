
import math

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
    return (distance + 9) // 10  # Ensure we round up

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
    nexus_locations = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
    start = (0, 0)
    # end = (10, 10)
    # Find the path using the greedy approach
    greedy_path = find_greedy_path(start, nexus_locations)
    print("Greedy Path:", greedy_path)
    journey_data, total_distance, total_packages, total_penalty, total_recovery = journey_details(greedy_path)
    print("Journey Details:", journey_data)
    d = 100  # Distance between start and end points
    # Calculate the score
    score = calculate_score(total_distance, d, total_packages, total_recovery, total_penalty)
    print("Score:", score)

# # Generate the output for nodes visited and packages dropped off
# output_content = []
# for data in journey_data:
#     line = f"From {data['start']} to {data['end']}: {data['packages']} packages"
#     output_content.append(line)

# # Save the output to a .txt file
# output_path = "expedition_output.txt"
# with open(output_path, "w") as file:
#     for line in output_content:
#         file.write(line + "\n")

# print("Output saved to expedition_output.txt")

# Generate the output in the desired format
output_content = ["["]
for data in journey_data:
    line = f"	[{data['packages']}, [{data['end']}]],"
    output_content.append(line)
output_content.append("]")

# Save the output to a .txt file
output_path = "expedition_output_format.txt"
with open(output_path, "w") as file:
    for line in output_content:
        file.write(line + "\n")

print("Output saved to expedition_output_format.txt")

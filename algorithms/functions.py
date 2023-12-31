import math
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


def compute_journey_info(start, path):
    """Compute the journey information: travel times, food requirements, weight penalties, and recovery times."""
    travel_times = []
    food_needs = []
    penalties = []
    recoveries = []
    
    current_position = start
    total_packages = 0
    
    for node in path:
        # Calculate travel time to the next node
        time = manhattan_distance(current_position, node)
        travel_times.append(time)
        
        # Calculate food required and add to total packages
        packages = food_required(time)
        food_needs.append(packages)
        total_packages += packages
        
        # Calculate weight penalty
        penalty = weight_penalty(time, packages)
        penalties.append(penalty)
        
        # Calculate recovery time for I.D.G.A.F.
        recovery = recovery_mins(packages)
        recoveries.append(recovery)
        
        current_position = node
    
    return travel_times, food_needs, penalties, recoveries

def output(journey_data):
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
        packages = food_required(distance)
        penalty = weight_penalty(distance, packages)
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
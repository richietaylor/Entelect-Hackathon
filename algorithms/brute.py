
import random
import math

def manhattan_distance(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def food_required(travel_time):
    out = travel_time // 10
    rem = travel_time % 10
    if rem >= 5:
        out = out + 1
    return out

def weight_penalty(travel_time, packages):
    penalty = 0
    remaining_time = travel_time
    while remaining_time > 0:
        penalty += packages
        remaining_time -= 10
        if remaining_time > 0:
            packages -= 1
    return penalty

def recovery_mins(travel_time):
    t = math.ceil(travel_time / 10)
    rem = travel_time % 10
    rec = 0
    if rem >= 5:
        rec = t * 10 - travel_time
    return rec

def compute_score(start, end, travel_times, packages_sent, recoveries, penalties):
    D = travel_times
    d = manhattan_distance(start, end)
    P = packages_sent
    T = recoveries
    W = penalties
    SF = 20 ** (math.log(0.1 * d * (D + 0.01) / d) + 0.8 ** P - 1.1 ** T + 10 / (1 + W))
    return round(SF)

def journey_details(path):
    total_distance = 0
    total_packages = 0
    total_penalty = 0
    total_recovery = 0
    for i in range(len(path) - 1):
        start, end = path[i], path[i+1]
        distance = manhattan_distance(start, end)
        packages = food_required(distance)
        penalty = weight_penalty(distance, packages)
        total_distance += distance
        total_packages += packages
        total_penalty += penalty
        total_recovery += recovery_mins(distance)
    return total_distance, total_packages, total_penalty, total_recovery

def objective_function(path):
    total_distance, total_packages, total_penalty, total_recovery = journey_details(path)
    return compute_score(start, path[-1], total_distance, total_packages, total_recovery, total_penalty)

def random_brute_force(initial_path, iterations=1000):
    best_path = initial_path
    best_score = objective_function(initial_path)
    for _ in range(iterations):
        random_path = [start] + random.sample(nexus_locations, len(nexus_locations)) + [end_point]
        current_score = objective_function(random_path)
        if current_score > best_score:
            best_score = current_score
            best_path = random_path
    return best_path, best_score

start = (0, 0)
end_point = (100, 100)
nexus_locations = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
initial_path = [(0, 0), (5, 68), (80, 66), (89, 74), (95, 60), (55, 96), (100, 100)]
best_path, best_score = random_brute_force(initial_path)

print("Best Path:", best_path)
print("Best Score:", best_score)
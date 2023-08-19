
import math
import random

# Calculate Manhattan distance between two nodes
def manhattan_distance(node1, node2):
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

# Calculate the number of food packages required for a given travel time
def food_required(travel_time):
    return math.ceil(travel_time / 10)

# Calculate the weight penalty for carrying a certain number of packages over a given travel time
def weight_penalty(packages, travel_time):
    penalty = 0
    for i in range(travel_time):
        if i % 10 == 0 and i != 0:
            packages -= 1
        penalty += packages
    return penalty

# Score function
def score(D, d, P, T, W):
    return 20 * (math.log(0.1 * d * (D + 0.01/d)) + 0.8 * P - 1.1 * T + 10 / (1 + W))

# Evaluate the fitness of a solution
def evaluate(solution, locations):
    total_distance = 0
    total_recovery = 0
    total_packages = 0
    total_weight_penalty = 0
    current_node = (0, 0)
    
    for package, loc in solution:
        travel_time = manhattan_distance(current_node, loc)
        total_distance += travel_time
        required_food = food_required(travel_time)
        total_packages += required_food
        total_recovery += required_food * 10
        penalty = weight_penalty(package, travel_time)
        total_weight_penalty += penalty
        
        current_node = loc
    
    # Calculate the final score
    d = manhattan_distance((0,0), locations[-1])
    s = score(total_distance, d, total_packages, total_recovery, total_weight_penalty)
    return s

# Simulated Annealing for the optimization problem
def simulated_annealing(locations):
    # Parameters for SA
    initial_temperature = 1000
    cooling_rate = 0.995
    min_temperature = 0.1
    
    # Generate an initial random solution
    current_solution = [[random.randint(1, 10), loc] for loc in locations[:-1]]
    current_score = evaluate(current_solution, locations)
    
    best_solution = current_solution
    best_score = current_score
    
    temperature = initial_temperature
    while temperature > min_temperature:
        # Generate a neighboring solution
        neighbor_solution = [s.copy() for s in current_solution]
        mutation_point = random.randint(0, len(neighbor_solution) - 1)
        if random.random() < 0.5:  # Change the number of packages
            neighbor_solution[mutation_point][0] = random.randint(1, 10)
        else:  # Change the location
            new_location = random.choice(locations[:-1])
            neighbor_solution[mutation_point][1] = new_location
        neighbor_score = evaluate(neighbor_solution, locations)
        
        # Decide whether to move to the neighboring solution
        if neighbor_score > current_score or random.random() < math.exp((neighbor_score - current_score) / temperature):
            current_solution = neighbor_solution
            current_score = neighbor_score
        
        # Update the best solution found so far
        if current_score > best_score:
            best_solution = current_solution
            best_score = current_score
        
        # Decrease the temperature
        temperature *= cooling_rate
    
    return best_solution, round(best_score)

# Example Usage
if __name__ == "__main__":
    locations = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
    solution, score_value = simulated_annealing(locations)
    print(f"Solution: {solution}")
    print(f"Score: {score_value}")

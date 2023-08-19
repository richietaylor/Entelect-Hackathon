def manhattan_distance(node1, node2):
    """Calculate the Manhattan distance between two nodes."""
    return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

def food_required(travel_time):
    """Calculate the number of food packages required given a travel time."""
    # If travel time is a multiple of 10, divide by 10
    # Otherwise, divide by 10 and round up
    return (travel_time + 9) // 10

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

# Providing the function for reference
weight_penalty


# Test the functions
test_node1 = (0, 0)
test_node2 = (3, 5)
test_travel_time = manhattan_distance(test_node1, test_node2)
test_food_needed = food_required(test_travel_time)

test_travel_time, test_food_needed

def greedy_path(start, end, nodes):
    """Find a path using a greedy approach, choosing the closest node to the current position."""
    path = []
    current_position = start
    remaining_nodes = nodes.copy()
    
    while remaining_nodes:
        # Find the closest node to the current position
        closest_node = min(remaining_nodes, key=lambda node: manhattan_distance(current_position, node))
        path.append(closest_node)
        remaining_nodes.remove(closest_node)
        current_position = closest_node

    # Add the end node to the path
    if end not in path:
        path.append(end)
    
    return path

# Test the function with an example set of nodes
test_nodes = [(55, 96), (95, 60), (5, 68), (80, 66), (89, 74), (100, 100)]
start_node = (0, 0)
end_node = (10, 10)
greedy_route = greedy_path(start_node, end_node, test_nodes)
greedy_route

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
        recovery = packages * 10
        recoveries.append(recovery)
        
        current_position = node
    
    return travel_times, food_needs, penalties, recoveries

# Compute journey information for the greedy path
travel_times_greedy, food_needs_greedy, penalties_greedy, recoveries_greedy = compute_journey_info(start_node, greedy_route)
travel_times_greedy, food_needs_greedy, penalties_greedy, recoveries_greedy

import math

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
    SF = 20 * (math.log(0.1 * d * (D + 0.01) / d) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    return round(SF)

# Compute the final score for the greedy path
greedy_score = compute_score(start_node, end_node, travel_times_greedy, food_needs_greedy, recoveries_greedy, penalties_greedy)
print("Greedy score: " + str(greedy_score))


import random

def initialize_population(nodes, population_size):
    """Generate an initial population of random paths."""
    population = []
    
    for _ in range(population_size):
        # Randomly shuffle the nodes to create a chromosome (path)
        chromosome = nodes.copy()
        random.shuffle(chromosome)
        population.append(chromosome)
    
    return population

# Test the initialization function
population_size = 10  # for example, create 10 random paths
initial_population = initialize_population(test_nodes, population_size)
initial_population

def evaluate_fitness(chromosome, start, end):
    """Evaluate the fitness of a chromosome (path)."""
    # Compute journey information for the chromosome
    travel_times, food_needs, penalties, recoveries = compute_journey_info(start, chromosome)
    
    # Calculate the score for the chromosome
    score = compute_score(start, end, travel_times, food_needs, recoveries, penalties)
    
    # Return the negative of the score as fitness (since we want to maximize the score)
    return -score

# Evaluate the fitness of the initial population
fitness_values = [evaluate_fitness(chromosome, start_node, end_node) for chromosome in initial_population]
fitness_values

def tournament_selection(population, fitness_values, tournament_size):
    """Select a chromosome using tournament selection."""
    # Select a few random chromosomes for the tournament
    selected_indices = random.sample(range(len(population)), tournament_size)
    tournament_chromosomes = [population[i] for i in selected_indices]
    tournament_fitness = [fitness_values[i] for i in selected_indices]
    
    # Select the best chromosome among the tournament chromosomes
    best_index = selected_indices[tournament_fitness.index(min(tournament_fitness))]  # Recall: lower fitness is better because we negated the score
    return population[best_index]

def create_new_population(population, fitness_values, tournament_size, population_size):
    """Create a new population using tournament selection."""
    new_population = []
    for _ in range(population_size):
        selected_chromosome = tournament_selection(population, fitness_values, tournament_size)
        new_population.append(selected_chromosome)
    return new_population

# Create a new population using tournament selection
tournament_size = 3
new_population = create_new_population(initial_population, fitness_values, tournament_size, population_size)
new_population

def ordered_crossover(parent1, parent2):
    """Perform ordered crossover to produce two offspring."""
    size = len(parent1)
    
    # Choose two random crossover points
    start, end = sorted(random.sample(range(size), 2))
    
    # Initialize offspring with None values
    offspring1 = [None] * size
    offspring2 = [None] * size
    
    # Copy the segment between crossover points from parents to offspring
    offspring1[start:end] = parent1[start:end]
    offspring2[start:end] = parent2[start:end]
    
    # Fill the remaining positions in the offspring
    for i in range(size):
        if offspring1[i] is None:
            for gene in parent2:
                if gene not in offspring1:
                    offspring1[i] = gene
                    break
        if offspring2[i] is None:
            for gene in parent1:
                if gene not in offspring2:
                    offspring2[i] = gene
                    break
                    
    return offspring1, offspring2

def mutate(chromosome):
    """Perform mutation by swapping two random nodes."""
    index1, index2 = random.sample(range(len(chromosome)), 2)
    mutated = chromosome.copy()
    mutated[index1], mutated[index2] = mutated[index2], mutated[index1]
    return mutated

# Test the genetic operators
parent1 = initial_population[0]
parent2 = initial_population[1]
offspring1, offspring2 = ordered_crossover(parent1, parent2)
mutated_offspring = mutate(offspring1)

offspring1, offspring2, mutated_offspring

def genetic_algorithm(nodes, start, end, population_size=50, generations=100, mutation_prob=0.1, tournament_size=3):
    """Genetic Algorithm for the inter-dimensional expedition problem."""
    
    # Initialize the population
    population = initialize_population(nodes, population_size)
    
    for generation in range(generations):
        new_population = []
        
        # Evaluate the fitness of the current population
        fitness_values = [evaluate_fitness(chromosome, start, end) for chromosome in population]
        
        # Create a new population
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)
            
            # Crossover
            offspring1, offspring2 = ordered_crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_prob:
                offspring1 = mutate(offspring1)
            if random.random() < mutation_prob:
                offspring2 = mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Replace old population with the new population
        population = new_population
    
    # Return the best solution from the last population
    best_chromosome = min(population, key=lambda chromosome: evaluate_fitness(chromosome, start, end))
    return best_chromosome

def genetic_algorithm_with_elitism(nodes, start, end, population_size=100, generations=100, mutation_prob=0.5, tournament_size=3, elitism_ratio=0.1):
    """Genetic Algorithm with Elitism for the inter-dimensional expedition problem."""
    
    # Initialize the population
    population = initialize_population(nodes, population_size)
    
    # Number of elite individuals
    num_elites = int(elitism_ratio * population_size)
    
    for generation in range(generations):
        new_population = []
        
        # Evaluate the fitness of the current population
        fitness_values = [evaluate_fitness(chromosome, start, end) for chromosome in population]
        
        # Elitism: Preserve the top individuals
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])
        elites = [population[i] for i in sorted_indices[:num_elites]]
        new_population.extend(elites)
        
        # Create the rest of the new population
        while len(new_population) < population_size:
            # Selection
            parent1 = tournament_selection(population, fitness_values, tournament_size)
            parent2 = tournament_selection(population, fitness_values, tournament_size)
            
            # Crossover
            offspring1, offspring2 = ordered_crossover(parent1, parent2)
            
            # Mutation
            if random.random() < mutation_prob:
                offspring1 = mutate(offspring1)
            if random.random() < mutation_prob:
                offspring2 = mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Replace old population with the new population
        population = new_population
    
    # Return the best solution from the last population
    best_chromosome = min(population, key=lambda chromosome: evaluate_fitness(chromosome, start, end))
    return best_chromosome

# Run the Genetic Algorithm with Elitism
best_path_elitism = genetic_algorithm_with_elitism(test_nodes, start_node, end_node)
best_path_elitism_score = compute_score(start_node, end_node, *compute_journey_info(start_node, best_path_elitism))

print("Elitism : " + str(best_path_elitism_score))

# Run the Genetic Algorithm
best_path = genetic_algorithm(test_nodes, start_node, end_node)
best_path



# Evaluate the score for the best path found by the genetic algorithm
best_path_score = compute_score(start_node, end_node, *compute_journey_info(start_node, best_path))
best_path_score
print(best_path_score)

def simulated_annealing(nodes, start, end, initial_temperature=5000, cooling_rate=0.995, max_iterations=10000, tabu_length=5):
    """Simulated Annealing algorithm for the inter-dimensional expedition problem with tabu list."""
    
    # Generate an initial solution randomly
    current_solution = nodes.copy()
    random.shuffle(current_solution)
    
    current_cost = evaluate_fitness(current_solution, start, end)
    
    best_solution = current_solution
    best_cost = current_cost
    
    temperature = initial_temperature
    
    # Initialize tabu list
    tabu_list = [current_solution.copy()]
    
    for iteration in range(max_iterations):
        # Generate a neighboring solution by swapping two random nodes
        neighbor = current_solution.copy()
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Check if neighbor is in tabu list
        while neighbor in tabu_list:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Calculate the cost of the neighboring solution
        neighbor_cost = evaluate_fitness(neighbor, start, end)
        
        # Calculate the cost difference
        cost_diff = neighbor_cost - current_cost
        
        # Check if the neighboring solution should be accepted
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution, current_cost = neighbor, neighbor_cost
            
            # Update the best solution if needed
            if current_cost < best_cost:
                best_solution, best_cost = current_solution, current_cost
            
            # Add current_solution to tabu list
            tabu_list.append(current_solution.copy())
            if len(tabu_list) > tabu_length:
                tabu_list.pop(0)  # remove the oldest solution
        
        # Reduce the temperature
        temperature *= cooling_rate
    
    return best_solution

# Let's test the modified simulated annealing function with a tabu list
test_nodes = [(7, 6), (3, 5), (9, 8)]
test_start = (0, 0)
test_end = (10, 10)

best_path_sa = simulated_annealing(test_nodes, test_start, test_end)
best_path_sa


# Run the Simulated Annealing algorithm
best_path_sa_score = compute_score(start_node, end_node, *compute_journey_info(start_node, best_path_sa))

best_path_sa, best_path_sa_score
print("Simulated Anneling : "+str(best_path_sa_score))
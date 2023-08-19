import numpy as np
import math


# Manhattan Distance
def manhattan_distance(point1, point2):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

# Calculate the required packages, recovery minutes, and weight penalties for a given journey
def compute_journey_info(start, path):
    current_location = start
    travel_times = []
    packages_required = []
    recovery_times = []
    weight_penalties = []

    for node in path:
        travel_time = manhattan_distance(current_location, node)
        packages_needed = math.ceil(travel_time / 10)
        recovery_time = packages_needed * 10
        weight_penalty = sum(packages_needed - i for i in range(packages_needed))

        travel_times.append(travel_time)
        packages_required.append(packages_needed)
        recovery_times.append(recovery_time)
        weight_penalties.append(weight_penalty)

        current_location = node

    return travel_times, packages_required, recovery_times, weight_penalties

# Compute the final score using the provided formula
def compute_score(start, end, travel_times, packages_sent, recoveries, penalties):
    D = sum(travel_times)
    d = manhattan_distance(start, end)
    P = sum(packages_sent)
    T = sum(recoveries)
    W = sum(penalties)
    SF = 20 ** (math.log(0.1 * d * (D + 0.01) / d) + 0.8 * P - 1.1 * T + 10 / (1 + W))
    return round(SF)

# Test nodes and starting and ending points
test_nodes = [(3, 5), (7, 6), (9, 8)]
start_node = (0, 0)
end_node = (10, 10)



class AntColonyOptimization:
    def __init__(self, nodes, start, end, ant_count=10, alpha=1, beta=2, evaporation_rate=0.5, pheromone_amount=100, max_iterations=100):
        # Ensure that start and end nodes are in the list
        if start not in nodes:
            nodes.insert(0, start)
        if end not in nodes:
            nodes.append(end)
            
        self.nodes = nodes
        self.start = start
        self.end = end
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_amount = pheromone_amount
        self.max_iterations = max_iterations
        
        self.pheromone = [[1 for _ in nodes] for _ in nodes]
    
    def choose_next_node(self, current_node, visited_nodes):
        """Choose the next node based on pheromone levels and heuristic."""
        unvisited_nodes = [node for node in self.nodes if node not in visited_nodes]
        
        if not unvisited_nodes:
            return self.end
        
        current_index = self.nodes.index(current_node)
        probabilities = []
        
        for node in unvisited_nodes:
            node_index = self.nodes.index(node)
            pheromone = self.pheromone[current_index][node_index] ** self.alpha
            heuristic = (1 / manhattan_distance(current_node, node)) ** self.beta
            probabilities.append(pheromone * heuristic)
        
        sum_probabilities = sum(probabilities)
        
        # Ensure that sum_probabilities is not 0 to avoid division by zero
        if sum_probabilities == 0:
            probabilities = [1/len(unvisited_nodes) for _ in unvisited_nodes]
        else:
            probabilities = [prob / sum_probabilities for prob in probabilities]
        
        # Check for negative probabilities and reset them to a small positive value
        probabilities = [max(prob, 1e-10) for prob in probabilities]
        
        # Normalize probabilities again to ensure they sum to 1
        sum_probabilities = sum(probabilities)
        probabilities = [prob / sum_probabilities for prob in probabilities]
        
        chosen_node = np.random.choice(len(unvisited_nodes), p=probabilities)
        return unvisited_nodes[chosen_node]

    def construct_solution(self):
        """Construct a solution (path) for one ant."""
        visited_nodes = [self.start]
        current_node = self.start
        
        while current_node != self.end:
            next_node = self.choose_next_node(current_node, visited_nodes)
            visited_nodes.append(next_node)
            current_node = next_node
        
        return visited_nodes
    
    def update_pheromone(self, paths):
        """Update the pheromone levels on the edges based on the paths taken by the ants."""
        for path in paths:
            for i in range(len(path) - 1):
                start_index = self.nodes.index(path[i])
                end_index = self.nodes.index(path[i+1])
                self.pheromone[start_index][end_index] += self.pheromone_amount / compute_score(self.start, self.end, *compute_journey_info(self.start, path))
                self.pheromone[end_index][start_index] += self.pheromone_amount / compute_score(self.start, self.end, *compute_journey_info(self.start, path))
        
        # Evaporate pheromone
        self.pheromone = [[(1 - self.evaporation_rate) * pheromone for pheromone in row] for row in self.pheromone]
    
    def optimize(self):
        best_path = None
        best_score = float('-inf')
        
        for iteration in range(self.max_iterations):
            paths = [self.construct_solution() for _ in range(self.ant_count)]
            self.update_pheromone(paths)
            
            for path in paths:
                score = compute_score(self.start, self.end, *compute_journey_info(self.start, path))
                if score > best_score:
                    best_score = score
                    best_path = path
        
        return best_path, best_score


aco_nodes = [start_node] + test_nodes + [end_node]

# Running the ACO algorithm
aco = AntColonyOptimization(test_nodes, start_node, end_node)
best_path_aco, best_score_aco = aco.optimize()

best_path_aco, best_score_aco

print(best_score_aco)
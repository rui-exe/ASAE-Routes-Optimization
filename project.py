import csv 
import numpy as np
import copy, random, math
import datetime
import pandas as pd

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"

def read_csv(filename):
    return pd.read_csv(filename)


#distances = np.loadtxt(distancesFileName, delimiter=",")
establishments = read_csv(establishmentsFileName)
#print second row
print(establishments.iloc[1]["Opening Hours"])

num_establishments = len(establishments) - 1
num_vehicles = math.floor(0.1*num_establishments)
max_hours = 8

END_OF_SHIFT = datetime.time(17, 0)


def can_visit(establishment, vehicle):
    current_establishment = vehicle["establishments"][-1]
    current_time = vehicle["current_time"]
    distance = distances[current_establishment][establishment]
    if  current_time + datetime.timedelta(minutes=distance) < END_OF_SHIFT:
        return True
    return None


def generate_random_solution():
    solution=[{"establishments":[],
               "current_time":datetime.time(9, 0),
              } for _ in range(0,num_vehicles)]
    print(solution)
    
    establishments_shuffled = random.shuffle(copy.deepcopy(establishments))

    for establishment in establishments_shuffled:
        vehicles_to_check = random.shuffle(list(range(0,num_vehicles)))
        for vehicle_to_check in vehicles_to_check:
            print()

def evaluate_solution(solution):
    return None

def get_neighbor_solution(solution):
    return None
    
    
def get_hc_solution(num_iterations, log=False):
    iteration = 0
    best_solution = generate_random_solution()
    best_score = evaluate_solution(best_solution)

    print(f"Init Solution:  {best_solution}, score: {best_score}")

    while iteration < num_iterations:
        iteration += 1
        neighbor = get_neighbor_solution(best_solution)
        if evaluate_solution(neighbor)>best_score:
            best_score = evaluate_solution(best_solution)
            best_solution = neighbor
            iteration=0 

        if log:
            (print(f"Solution:       {best_solution}, score: {best_score}"))

    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution


def get_sa_solution(num_iterations, log=False):
    iteration = 0
    temperature = 1000000
    solution = generate_random_solution() # Best solution after 'num_iterations' iterations without improvement
    score = evaluate_solution(solution)

    best_solution = copy.deepcopy(solution)
    best_score = score

    print(f"Init Solution:  {best_solution}, score: {best_score}")

    while iteration < num_iterations:
        temperature = temperature * 0.999  # Test with different cooling schedules
        iteration += 1

        neighbor = get_neighbor_solution(solution)
        neighbor_score=evaluate_solution(neighbor)
        delta_e = neighbor_score-score
        if delta_e>0 or np.exp(delta_e/temperature)>random.random():
            score = neighbor_score
            solution = neighbor 

            if(score>best_score):
                iteration=1
                best_score=score
                best_solution=solution

        if log:
            print(f"Solution: {best_solution}, score: {best_score},  Temp: {temperature}")

    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution


# Genetic Algorithms 

# 4.2 c)
def midpoint_crossover(solution_1, solution_2):
    return None

def randompoint_crossover(solution_1, solution_2):
    return None

#4.2 d)
def generate_population(population_size):
    solutions = []
    for i in range(population_size):
        solutions.append(generate_random_solution())
    return solutions

def print_population(population):
    solutions = []
    for i in range(len(population)):
        print(f"Solution {i}: {population[i]}, {evaluate_solution(population[i])}")
    
def tournament_select(population, tournament_size):
    competing_elements = [population[index] for index in np.random.choice(len(population), tournament_size)]
    return max(competing_elements,key=evaluate_solution)

def get_greatest_fit(population):
    best_solution = population[0]
    best_score = evaluate_solution(population[0])
    for i in range(1, len(population)):
        score = evaluate_solution(population[i])
        if score > best_score:
            best_score = score
            best_solution = population[i]
    return best_solution, best_score

def replace_least_fittest(population, offspring):
    least_fittest_index = 0
    least_fittest_value = evaluate_solution(population[0])
    for i in range(1, len(population)):
        score = evaluate_solution(population[i])
        if score < least_fittest_value:
            least_fittest_value = score
            least_fittest_index = i
    population[least_fittest_index] = offspring

def roulette_select(population):
    worst_solution = min(evaluate_solution(solution) for solution in population) 
    total = sum(evaluate_solution(solution)-worst_solution+1 for solution in population)   
    random_number = random.random()
    current_percentage=0
    for solution in population:
        lower_bound = current_percentage
        upper_bound = current_percentage+(evaluate_solution(solution)-worst_solution+1)/total
        if random_number>=lower_bound and random_number<=upper_bound:
            return solution
        else:
            current_percentage+=(evaluate_solution(solution)-worst_solution+1)/total


def mutate_solution(solution):
    return solution


def genetic_algorithm(num_iterations, population_size, crossover_func, mutation_func, log=False):
    population = generate_population(population_size)
    
    best_solution = population[0] # Initial solution
    best_score = evaluate_solution(population[0])
    best_solution_generation = 0 # Generation on which the best solution was found
    
    generation_no = 0
    
    print(f"Initial solution: {best_solution}, score: {best_score}")
    
    while(num_iterations > 0):
        
        generation_no += 1
        
        tournment_winner_sol = random.choice(population)
        roulette_winner_sol = roulette_select(population)
        
        (child1,child2) = crossover_func(tournment_winner_sol,roulette_winner_sol)
        
        for child in [child1,child2]:
            if(random.random()<0.03):
                child = mutation_func(child)
            replace_least_fittest(population,child)
        
        
        # Checking the greatest fit among the current population
        greatest_fit, greatest_fit_score = get_greatest_fit(population)
        if greatest_fit_score > best_score:
            best_solution = greatest_fit
            best_score = greatest_fit_score
            best_solution_generation = generation_no
            if log:
                print(f"\nGeneration: {generation_no }")
                print(f"Solution: {best_solution}, score: {best_score}")
                print_population(population)
        else:
            num_iterations -= 1
        
    print(f"  Final solution: {best_solution}, score: {best_score}")
    print(f"  Found on generation {best_solution_generation}")
    
    return best_solution
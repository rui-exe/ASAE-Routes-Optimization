import numpy as np
import random
import datetime
import utils
import time
import matplotlib.pyplot as plt
import datetime
from neighborhood_genetic import get_neighbor_solution

# Genetic Algorithms 

"""
    Legalize Solution
    Parameters:
        solution_1_establishments: establishments of the first solution
        solution_2_establishments: establishments of the second solution
        unchanged_solution_establishments: establishments of the unchanged solution
        maps: maps used in the partially mapped crossover
        lower_point: lower point of the crossover
        upper_point: upper point of the crossover
    Returns:
        solution_1_establishments: legalized establishments of the first solution
"""	
def legalize_solution(solution_1_establishments, solution_2_establishments,unchanged_solution_establishments, maps, lower_point, upper_point):
    for idx in range(len(solution_1_establishments)):
        if idx < lower_point or idx > upper_point:
            for s in maps:
                if solution_1_establishments[idx] in s:
                    for establishment in s:
                        if establishment not in unchanged_solution_establishments:
                            unchanged_solution_establishments.remove(solution_1_establishments[idx])
                            solution_1_establishments[idx] = establishment
                            unchanged_solution_establishments.add(establishment)
                            break
                    break
        else:
            solution_1_establishments[idx] = solution_2_establishments[idx]
    return solution_1_establishments

"""
    Populate Vehicle with Establishments
    Parameters:
        univisited_establishments: unvisited establishments
    Returns:
        vehicle: vehicle with establishments
        univisited_establishments: unvisited establishments that were not added to the vehicle
"""

def populate_vehicle(univisited_establishments):
    vehicle = {"establishments":[],
               "current_time":datetime.time(9, 0),
              }
    for establishment in univisited_establishments:
        end_of_inspection = utils.can_visit(vehicle,establishment)
        if end_of_inspection:
            vehicle["establishments"].append(establishment)
            vehicle["current_time"] = end_of_inspection
        else:
            break
    for establishment in vehicle["establishments"]:
        univisited_establishments.remove(establishment)
    return vehicle, univisited_establishments

"""
    Populate Solution with Establishments
    Parameters: solution_1_establishments: establishments of the first solution
    Returns: solution: solution with establishments
"""
def populate_solution(solution_1_establishments):
    solution = {"vehicles":[{"establishments":[],
               "current_time":datetime.time(9, 0),
              } for _ in range(0,utils.num_vehicles)],
              "unvisited_establishments":list(range(1,utils.num_establishments+1))}


    for establishment in solution_1_establishments:
        vehicles_to_check = list(range(0,utils.num_vehicles))
        for vehicle_to_check in vehicles_to_check:
            end_of_inspection = utils.can_visit(solution["vehicles"][vehicle_to_check],establishment)
            if end_of_inspection:
                solution["vehicles"][vehicle_to_check]["establishments"].append(establishment)
                solution["unvisited_establishments"].remove(establishment)
                solution["vehicles"][vehicle_to_check]["current_time"] = end_of_inspection
                break
    for vehicle in solution["vehicles"]:
        if vehicle["establishments"] == []:
            vehicle, solution["unvisited_establishments"] = populate_vehicle(solution["unvisited_establishments"])
        time_to_depot = utils.distances.loc[f'p_{vehicle["establishments"][-1]}']['p_0']
        vehicle["current_time"] = utils.add_seconds(vehicle["current_time"],time_to_depot)

    return solution

"""
    Partially Matched Crossover
    Parameters: solution_1: first solution to be crossed
                solution_2: second solution to be crossed
    Returns:    child_1: first child after crossover
                child_2: second child after crossover
"""

def final_crossover(solution_1, solution_2):
    solution_1_establishments = []
    solution_2_establishments = []
    for vehicle in solution_1["vehicles"]: 
        solution_1_establishments += vehicle["establishments"]

    for vehicle in solution_2["vehicles"]:
        solution_2_establishments += vehicle["establishments"]


    min_size = min(len(solution_1_establishments),len(solution_2_establishments)) - 1
    random_point1 = random.randint(0,min_size)
    random_point2 = random.randint(0,min_size)
    while(random_point2==random_point1):
        random_point2 = random.randint(0,min_size)

    lower_point, upper_point = (random_point1, random_point2) if random_point1 < random_point2 else (random_point2, random_point1)

    maps = []
    for idx in range(lower_point, upper_point+1):
        present = False
        for s in maps:
            if solution_1_establishments[idx] in s:
                s.add(solution_2_establishments[idx])
                present = True
                break
            elif solution_2_establishments[idx] in s:
                s.add(solution_1_establishments[idx])
                present = True
                break
        if not present:
                maps.append(set([solution_1_establishments[idx], solution_2_establishments[idx]]))

    joined_sets = []

    while len(maps) > 0:
        current_set = maps.pop(0)
        joined = False
        
        for i in range(len(maps)):
            if current_set.intersection(maps[i]):
                current_set |= maps.pop(i)
                joined = True
                break
                
        if not joined:
            joined_sets.append(current_set)
        else:
            maps.append(current_set)
            
    # add any remaining sets to joined_sets
    for s in maps:
        joined_sets.append(s)
            

    unchanged_solution_1_establishments = set(solution_1_establishments[:lower_point] + solution_1_establishments[upper_point+1:] + solution_2_establishments[lower_point:upper_point+1])
    unchanged_solution_2_establishments = set(solution_2_establishments[:lower_point] + solution_2_establishments[upper_point+1:] + solution_1_establishments[lower_point:upper_point+1])


    solution_1_establishments_copy = solution_1_establishments.copy()

    
    solution_1_establishments = legalize_solution(solution_1_establishments, solution_2_establishments,unchanged_solution_1_establishments, joined_sets, lower_point, upper_point)
    solution_2_establishments = legalize_solution(solution_2_establishments, solution_1_establishments_copy,unchanged_solution_2_establishments, joined_sets, lower_point, upper_point)

    
    child1 = populate_solution(solution_1_establishments)
    child2 = populate_solution(solution_2_establishments)

    return child1, child2

"""
    Generate the population
    Parameters: population_size: size of the population
    Returns:    solutions: list of solutions
    
"""
def generate_population(population_size):
    solutions = []
    for i in range(population_size):
        print(i)
        solutions.append(utils.generate_random_solution())
    return solutions

"""
    Print the population
    Parameters: population: population to be printed
"""
def print_population(population):
    solutions = []
    for i in range(len(population)):
        print(f"Solution {i}: {population[i]}, {utils.evaluate_solution(population[i])}")

"""
    Select the best solution from the population using tournament selection
    Parameters: population: population to be selected from
                tournament_size: size of the tournament
    Returns:    max(competing_elements,key=utils.evaluate_solution):  which is the best solution
                
"""

def tournament_select(population, tournament_size):
    competing_elements = [population[index] for index in np.random.choice(len(population), tournament_size)]
    return max(competing_elements,key=utils.evaluate_solution)
"""
    Select the best solution from the population
    Parameters: population: population to be selected from
"""
def get_greatest_fit(population):
    best_solution = population[0]
    best_score = utils.evaluate_solution(population[0])
    for i in range(1, len(population)):
        score = utils.evaluate_solution(population[i])
        if score > best_score:
            best_score = score
            best_solution = population[i]
    return best_solution, best_score

"""
    Replace the least fittest solution from the population
    Parameters: population: population to be replaced from
                offspring: offspring to be inserted
"""
def replace_least_fittest(population, offspring):
    least_fittest_index = 0
    least_fittest_value = utils.evaluate_solution(population[0])
    for i in range(1, len(population)):
        score = utils.evaluate_solution(population[i])
        if score < least_fittest_value:
            least_fittest_value = score
            least_fittest_index = i
    population[least_fittest_index] = offspring

"""
    Roulette wheel selection used in the genetic algorithm
    Parameters: population: population to be selected from
    Returns:    solution: selected solution
"""
def roulette_select(population):
    worst_solution = min(utils.evaluate_solution(solution) for solution in population) 
    total = sum(utils.evaluate_solution(solution)-worst_solution+1 for solution in population)   
    random_number = random.random()
    current_percentage=0
    for solution in population:
        lower_bound = current_percentage
        upper_bound = current_percentage+(utils.evaluate_solution(solution)-worst_solution+1)/total
        if random_number>=lower_bound and random_number<=upper_bound:
            return solution
        else:
            current_percentage+=(utils.evaluate_solution(solution)-worst_solution+1)/total

"""
    Mutation function
    Parameters: solution: solution to be mutated
    Returns:    get_neighbor_solution(solution): neighbor solution
"""
def mutate_solution(solution):
    return get_neighbor_solution(solution)

"""
    Genetic algorithm
    Parameters: num_iterations: number of iterations
                population_size: size of the population
                crossover_func: crossover function
                mutation_func: mutation function
                log: log the results
    Returns:    best_solution: best solution found
"""
def genetic_algorithm(num_iterations, population_size, crossover_func, mutation_func, log=False):
    population = generate_population(population_size)


    best_solution = max(population,key=utils.evaluate_solution) # Initial solution
    best_score = utils.evaluate_solution(best_solution)
    print(f"Initial best solution: {best_solution}")
    print(f"Initial best score: {best_score}")
    best_solution_generation = 0 # Generation on which the best solution was found
    
    generation_no = 0
    start_time = time.time()
    solution_utilities = [best_score]
    times = [0]
    
    
    while(num_iterations > 0):
        
        generation_no += 1
        #print(f"Generation {generation_no}")
        random_winner_sol = random.choice(population)
        tournment_winner_sol = tournament_select(population,5)
        roulette_winner_sol = roulette_select(population)
        
        
        child_1, child_2 = crossover_func(random_winner_sol,tournment_winner_sol)   
        child_1 = mutation_func(child_1)
        child_2 = mutation_func(child_2)
        
        replace_least_fittest(population,child_1)
        replace_least_fittest(population,child_2)

        
        # Checking the greatest fit among the current population
        greatest_fit, greatest_fit_score = get_greatest_fit(population)
        if greatest_fit_score > best_score:
            best_solution = greatest_fit
            best_score = greatest_fit_score
            best_solution_generation = generation_no
            solution_utilities.append(best_score)
            times.append(time.time()-start_time)
            if log:
                print(f"\nGeneration: {generation_no }")
                print(f"score: {best_score}")
        else:
            num_iterations -= 1
        
    print(f"  Final solution score: {best_score}")
    print(f"  Found on generation {best_solution_generation}")
    plt.plot(times, solution_utilities)
    plt.title('Solution Utility over Time (GA) - ' + str(utils.num_establishments) + ' Establishments')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Solution Utility')
    date = datetime.datetime.now()
    plt.savefig('plots/ga_solution_utility' + date.strftime("%Y%m%d-%H%M%S") + '.png')
    return best_solution

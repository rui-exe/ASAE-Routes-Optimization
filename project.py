import numpy as np
import copy, random, math
import datetime
import pandas as pd
import ast

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


#num_establishments = len(establishments) - 1
num_establishments = 30

num_vehicles = math.floor(0.1*num_establishments)
max_hours = 8

END_OF_SHIFT = datetime.time(17, 0)

def add_seconds(time, seconds):
    return (datetime.datetime.combine(datetime.datetime.today(), time) + datetime.timedelta(seconds=seconds)).time()
    

def add_minutes(time, minutes):
    return (datetime.datetime.combine(datetime.datetime.today(), time) + datetime.timedelta(minutes=minutes)).time()

def can_visit(vehicle,establishment):
    if(len(vehicle["establishments"])==0):
        time_to_establishment = distances.loc['p_0'][f'p_{establishment}']
    else:
        current_establishment = vehicle["establishments"][-1]
        time_to_establishment = distances.loc[f'p_{current_establishment}'][f'p_{establishment}']

    time_to_depot = distances.loc[f'p_{establishment}']['p_0']
    establishment_opening_hours = establishments.iloc[establishment]["Opening Hours"] # Get list with the working hours of the establishment
    inspection_duration = establishments.iloc[establishment]["Inspection Time"].item()
    current_time = vehicle["current_time"]


    arriving_time = add_seconds(current_time,time_to_establishment) # Add distance to current time
    end_of_inpection = add_minutes(arriving_time,inspection_duration) #Add inspection time to arriving time
    arrival_at_depot = add_seconds(end_of_inpection,time_to_depot) #Add distance to current time

    
    if arrival_at_depot < END_OF_SHIFT and establishment_opening_hours[arriving_time.hour]:
        return end_of_inpection
    else:
        return False
    
def generate_random_solution():
    solution={"vehicles":[{"establishments":[],
               "current_time":datetime.time(9, 0),
              } for _ in range(0,num_vehicles)],
              "unvisited_establishments":list(range(1,num_establishments))}
    
    establishments_shuffled = list(range(1,num_establishments))
    random.shuffle(establishments_shuffled)

    for establishment in establishments_shuffled:
        vehicles_to_check = list(range(0,num_vehicles))
        random.shuffle(vehicles_to_check)
        for vehicle_to_check in vehicles_to_check:
            end_of_inpection = can_visit(solution["vehicles"][vehicle_to_check],establishment)
            if end_of_inpection:
                solution["vehicles"][vehicle_to_check]["establishments"].append(establishment)
                solution["unvisited_establishments"].remove(establishment)
                solution["vehicles"][vehicle_to_check]["current_time"] = end_of_inpection
                break

    for vehicle in solution["vehicles"]:
        time_to_depot = distances.loc[f'p_{vehicle["establishments"][-1]}']['p_0']
        vehicle["current_time"] = add_seconds(vehicle["current_time"],time_to_depot)
        
    return solution

def minutes_not_working(vehicle):
    datetime1 = datetime.datetime.combine(datetime.date.today(), END_OF_SHIFT)
    datetime2 = datetime.datetime.combine(datetime.date.today(), vehicle["current_time"])

    timedelta = datetime1 - datetime2

    return timedelta.total_seconds() / 60

def evaluate_solution(solution):
    visited_establishments = num_establishments-len(solution["unvisited_establishments"])
    return visited_establishments



def is_possible(establishments):
    vehicle={"establishments":[],
               "current_time":datetime.time(9, 0),
              } 
    
    for idx,establishment in enumerate(establishments):
        end_of_inpection = can_visit(vehicle,establishment)
        if end_of_inpection:
            vehicle["establishments"].append(establishment)
            vehicle["current_time"] = end_of_inpection
        else:
            return (False,idx)

    if vehicle["establishments"]:
        time_to_depot = distances.loc[f'p_{vehicle["establishments"][-1]}']['p_0']
        vehicle["current_time"] = add_seconds(vehicle["current_time"],time_to_depot)
    
    return (True,vehicle)


def change_two_establishments_in_vehicle(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"]) == 0:
        return solution
    establishment_1 = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    establishments_to_visit = list(range(0,len(neighbor["vehicles"][vehicle]["establishments"])-1))
    random.shuffle(establishments_to_visit)
    for establishment_2 in establishments_to_visit:
        if establishment_1 != establishment_2:
            neighbor["vehicles"][vehicle]["establishments"][establishment_1],neighbor["vehicles"][vehicle]["establishments"][establishment_2] = neighbor["vehicles"][vehicle]["establishments"][establishment_2],neighbor["vehicles"][vehicle]["establishments"][establishment_1]
            (is_vehicle_possible,new_vehicle)=is_possible(neighbor["vehicles"][vehicle]["establishments"])
            if is_vehicle_possible:
                neighbor["vehicles"][vehicle]=new_vehicle
                return neighbor
            else:
                neighbor["vehicles"][vehicle]["establishments"][establishment_1],neighbor["vehicles"][vehicle]["establishments"][establishment_2] = neighbor["vehicles"][vehicle]["establishments"][establishment_2],neighbor["vehicles"][vehicle]["establishments"][establishment_1] # Undo change
            
    return solution

def change_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"])>1:
        establishment_1_index = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    elif len(neighbor["vehicles"][vehicle]["establishments"])==1:
        establishment_1_index = 0
    else:
        return neighbor
    establishment_to_mark_as_unvisited =  neighbor["vehicles"][vehicle]["establishments"][establishment_1_index]
    establishments_to_visit = copy.deepcopy(neighbor["unvisited_establishments"])
    random.shuffle(establishments_to_visit)
    
    for establishment_to_visit in establishments_to_visit:
        neighbor["unvisited_establishments"].remove(establishment_to_visit)
        neighbor["unvisited_establishments"].append(establishment_to_mark_as_unvisited)
        neighbor["vehicles"][vehicle]["establishments"][establishment_1_index] = establishment_to_visit

        (is_vehicle_possible,new_vehicle)=is_possible(neighbor["vehicles"][vehicle]["establishments"])
        if is_vehicle_possible:
            neighbor["vehicles"][vehicle]=new_vehicle
            return neighbor
        else:
            neighbor["unvisited_establishments"].append(establishment_to_visit)
            neighbor["unvisited_establishments"].remove(establishment_to_mark_as_unvisited)
            neighbor["vehicles"][vehicle]["establishments"][establishment_1_index] = establishment_to_mark_as_unvisited

    return solution

def calculate_current_time(vehicle):
    current_time = datetime.time(9, 0)
    establishments = range(1,len(vehicle["establishments"]))
    for establishment in establishments:
        current_time = add_seconds(current_time,distances.loc[f'p_{vehicle["establishments"][establishment-1]}'][f'p_{vehicle["establishments"][establishment]}'])
    return current_time

def remove_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"])>1:
        establishment_index = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    elif len(neighbor["vehicles"][vehicle]["establishments"])==1:
        establishment_index = 0
    else:
        return neighbor
    establishment_to_mark_as_unvisited =  neighbor["vehicles"][vehicle]["establishments"][establishment_index]
    neighbor["unvisited_establishments"].append(establishment_to_mark_as_unvisited)
    neighbor["vehicles"][vehicle]["establishments"].pop(establishment_index)
    neighbor["vehicles"][vehicle]["current_time"] = calculate_current_time(neighbor["vehicles"][vehicle])
    return neighbor
    

def add_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,num_vehicles-1)
    establishments_to_visit = copy.deepcopy(neighbor["unvisited_establishments"])
    random.shuffle(establishments_to_visit)
    
    for establishment_to_visit in establishments_to_visit:
        neighbor["unvisited_establishments"].remove(establishment_to_visit)
        neighbor["vehicles"][vehicle]["establishments"].append(establishment_to_visit)

        (is_vehicle_possible,new_vehicle)=is_possible(neighbor["vehicles"][vehicle]["establishments"])
        if is_vehicle_possible:
            neighbor["vehicles"][vehicle]=new_vehicle
            return neighbor
        else:
            neighbor["unvisited_establishments"].append(establishment_to_visit)
            neighbor["vehicles"][vehicle]["establishments"].remove(establishment_to_visit)
    return solution

def get_neighbor_solution(solution):
    neighbor_function = random.choice([change_two_establishments_in_vehicle,remove_random_establishment,change_random_establishment,add_random_establishment])
    return neighbor_function(solution)
    
def get_hc_solution(num_iterations, log=False):
    iteration = 0
    best_solution = generate_random_solution()
    best_score = evaluate_solution(best_solution)

    establishments_visited = num_establishments-len(best_solution["unvisited_establishments"])

    while iteration < num_iterations:
        iteration += 1
        neighbor = get_neighbor_solution(best_solution)
        if evaluate_solution(neighbor)>best_score:
            best_score = evaluate_solution(best_solution)
            best_solution = neighbor
            iteration=0 

        if log:
            (print(f"Solution:       {best_solution}, score: {best_score}"))

    print(f"Random Solution Score: {establishments_visited}")
    establishments_visited = num_establishments-len(best_solution["unvisited_establishments"])
    print(f"Final Solution Score: {establishments_visited}")
    return best_solution



#get_hc_solution(1000)



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
def lox_crossover(solution_1, solution_2):
                
    children = [dict(),dict()]
    for child_idx in range(2):
        vehicles_1 = solution_1["vehicles"]
        vehicles_2 = solution_2["vehicles"]
        unvisited_establishments = list(set(solution_1["unvisited_establishments"]).intersection(set(solution_2["unvisited_establishments"])))

        establishments_different_vehicle = dict() #establishment in solution 1 xor in solution 2 =>map[establishment]=False
                                                #else map[establishment]=True
        for (vehicle_1,vehicle_2) in zip(vehicles_1,vehicles_2):
            for establishment in vehicle_1["establishments"]:
                if establishment not in vehicle_2["establishments"]:
                    if establishment not in establishments_different_vehicle:
                        establishments_different_vehicle[establishment]=False #establishment only in one of the solutions
                    else:
                        establishments_different_vehicle[establishment]=True #establishment in both solutuins

            for establishment in vehicle_2["establishments"]:
                if establishment not in vehicle_1["establishments"]:
                    if establishment not in establishments_different_vehicle:
                        establishments_different_vehicle[establishment]=False
                    else:
                        establishments_different_vehicle[establishment]=True
            children[child_idx]["vehicles"]=[]
        
        for vehicle_idx in range(num_vehicles):
            vehicle1_establishments = vehicles_1[vehicle_idx]["establishments"]
            vehicle2_establishments = vehicles_2[vehicle_idx]["establishments"]
            common_establishments = [establishment for establishment in vehicle1_establishments if establishment in vehicle2_establishments]
            non_common_establishments = [establishment for establishment in vehicle1_establishments if establishment not in vehicle2_establishments]
            non_common_establishments += [establishment for establishment in vehicle2_establishments if establishment not in vehicle1_establishments]
            vehicle_establishments = []
            random.shuffle(common_establishments)
            random.shuffle(non_common_establishments)

            for establishment in common_establishments:
                vehicle_establishments.append(establishment)

            for establishment in non_common_establishments:
                if establishment in establishments_different_vehicle:
                    if(establishments_different_vehicle[establishment]):
                        if(random.random()<0.5):
                            vehicle_establishments.append(establishment)
                            del establishments_different_vehicle[establishment]
                        else:
                            establishments_different_vehicle[establishment]=False
                    else:
                        vehicle_establishments.append(establishment)
                        del establishments_different_vehicle[establishment]


            (is_vehicle_possible,new_vehicle) = is_possible(vehicle_establishments)
            while not is_vehicle_possible:
                failed_vehicle_idx = new_vehicle
                establishments_different_vehicle[vehicle_establishments[failed_vehicle_idx]]=False #Add it back to the map, it can now be in another vehicle
                vehicle_establishments = vehicle_establishments[:failed_vehicle_idx] + vehicle_establishments[failed_vehicle_idx+1:]
                (is_vehicle_possible,new_vehicle) = is_possible(vehicle_establishments)
            
            children[child_idx]["vehicles"].append(new_vehicle)

        children[child_idx]["unvisited_establishments"] = [establishment for establishment in establishments_different_vehicle] + unvisited_establishments
            

    [child_1,child_2]=children
    return child_1,child_2

#4.2 d)
def generate_population(population_size):
    solutions = []
    for i in range(population_size):
        print(i)
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
    return get_neighbor_solution(solution)


def genetic_algorithm(num_iterations, population_size, crossover_func, mutation_func, log=False):
    population = generate_population(population_size)
    
    best_solution = population[0] # Initial solution
    best_score = evaluate_solution(population[0])
    print(f"Initial best score: {max(population,key=evaluate_solution)}")
    best_solution_generation = 0 # Generation on which the best solution was found
    
    generation_no = 0
    
    print(f"Initial solution: {best_solution}, score: {best_score}")
    
    while(num_iterations > 0):
        
        generation_no += 1
        print(f"Generation {generation_no}")

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
        
    print(f"  Final solution score: {best_score}")
    print(f"  Found on generation {best_solution_generation}")
    
    return best_solution

#print(establishments["Inspection Time"].mean())

best_solution = genetic_algorithm(500, 50, lox_crossover, mutate_solution)
print(best_solution) 
"""

 solution_1 = generate_random_solution()
solution_2 = generate_random_solution()

print(solution_1)
print()
print(solution_2)
print()
print()
print()
(child_1,child_2)= lox_crossover(solution_1,solution_2)
print(child_1)
print()
print(child_2) """
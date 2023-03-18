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
num_establishments = 200

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
    while(not establishment_opening_hours[arriving_time.hour]):
        if(arriving_time.hour+1>=17):
            return False
        arriving_time = datetime.time(arriving_time.hour+1, 0)


    end_of_inpection = add_minutes(arriving_time,inspection_duration) #Add inspection time to arriving time
    arrival_at_depot = add_seconds(end_of_inpection,time_to_depot) #Add distance to current time

    
    if arrival_at_depot < END_OF_SHIFT:
        return end_of_inpection
    else:
        return False
    
def generate_random_solution():
    solution={"vehicles":[{"establishments":[],
               "current_time":datetime.time(9, 0),
              } for _ in range(0,num_vehicles)],
              "unvisited_establishments":list(range(1,num_establishments+1))}
    
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

def two_opt_operator(solution):
    # Choose two consecutive sublists
    neighbor = copy.deepcopy(solution)
    route_index = random.randint(0, len(neighbor["vehicles"]) - 1)
    route = neighbor["vehicles"][route_index]
    while len(route["establishments"]) < 3:
        route_index = random.randint(0, len(neighbor["vehicles"]) - 1)
        route = neighbor["vehicles"][route_index]
    sublist_index = random.randint(0, len(route["establishments"]) - 2)
    sublist1 = route["establishments"][sublist_index:sublist_index + 2]

    # Choose a sub-operator randomly
    sub_operator = random.choice([exchange_sublists_within_route, exchange_sublists_between_routes])

    # Apply the chosen sub-operator
    sub_operator(neighbor, route_index, sublist_index, sublist1)
    return neighbor

def exchange_sublists_within_route(solution, route_index, sublist_index, sublist1):
    # Choose another sublist in the same route
    if len(solution["vehicles"][route_index]["establishments"]) < 5:
        return solution
    sublist2_index = random.randint(0, len(solution["vehicles"][route_index]["establishments"]) - 2)
    while abs(sublist2_index - sublist_index) <= 1:
        sublist2_index = random.randint(0, len(solution["vehicles"][route_index]["establishments"]) - 2)
    sublist2 = solution["vehicles"][route_index]["establishments"][sublist2_index:sublist2_index + 2]

    # Exchange the two sublists
    solution["vehicles"][route_index]["establishments"][sublist_index:sublist_index + 2] = sublist2
    solution["vehicles"][route_index]["establishments"][sublist2_index:sublist2_index + 2] = sublist1
    return solution

def exchange_sublists_between_routes(solution, route_index, sublist_index, sublist1):
    # Choose another route
    route2_index = random.randint(0, len(solution["vehicles"]) - 1)
    while route2_index == route_index or len(solution["vehicles"][route2_index]["establishments"]) < 2:
        route2_index = random.randint(0, len(solution["vehicles"]) - 1)
    route2 = solution["vehicles"][route2_index]

    # Choose a sublist in the other route
    sublist2_index = random.randint(0, len(route2["establishments"]) - 2)
    sublist2 = route2["establishments"][sublist2_index:sublist2_index + 2]

    # Exchange the two sublists
    solution["vehicles"][route_index]["establishments"][sublist_index:sublist_index + 2] = sublist2
    route2["establishments"][sublist2_index:sublist2_index + 2] = sublist1
    return solution


def get_neighbor_solution(solution):
    neighbor_function = random.choice([add_random_establishment, remove_random_establishment, change_random_establishment, change_two_establishments_in_vehicle, two_opt_operator])
    print(neighbor_function)
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
    solution_1_establishments = []
    solution_2_establishments = []
    child = dict()
    unvisited_establishments = set([establishment for establishment in range(1,num_establishments+1)])


    for vehicle in solution_1["vehicles"]: 
        solution_1_establishments += vehicle["establishments"]

    for vehicle in solution_2["vehicles"]:
        solution_2_establishments += vehicle["establishments"]

    random_point1 = random.randint(0,len(solution_1_establishments))
    random_point2 = random.randint(0,len(solution_1_establishments))
    while(random_point2==random_point1):
        random_point2 = random.randint(0,len(solution_1_establishments))

    lower_point, upper_point = (random_point1, random_point2) if random_point1 < random_point2 else (random_point2, random_point1)
    #print((lower_point,upper_point))
    child_establishments = [0 for i in range(0,max(len(solution_1_establishments),len(solution_2_establishments)))]


    solution_1_chromossomes = solution_1_establishments[lower_point:upper_point+1]
    solution_2_chromossomes = [establishment for establishment in solution_2_establishments if establishment not in solution_1_chromossomes]
    
    child_establishments[lower_point:upper_point+1] = solution_1_chromossomes
    child_establishments[:lower_point] = solution_2_chromossomes[:lower_point]

    if(len(solution_1_establishments)<len(solution_2_establishments)):
        child_establishments + [0 for i in range(len(solution_2_establishments)-len(solution_1_establishments))]

    child_establishments[upper_point+1:] = solution_2_chromossomes[lower_point:]
        
    if(len(solution_1_establishments)>len(solution_2_establishments)):
        child_establishments + [0 for i in range(len(solution_1_establishments)-len(solution_2_establishments))]
        leftover_solution_1_chromossomes = [establishment for establishment in solution_1_establishments if establishment not in child_establishments]
        child_establishments[len(solution_2_establishments)+1:] = leftover_solution_1_chromossomes



    #seperate child into vehicles

    lower_index = 0
    upper_index = 0
    child["vehicles"] = []
    while upper_index < len(child_establishments) and len(child["vehicles"])<num_vehicles:
        lower_index = upper_index
        is_vehicle_possible, vehicle = is_possible(child_establishments[lower_index:upper_index])
        last_possible_vehicle = vehicle
        while upper_index < len(child_establishments):
            last_possible_vehicle = vehicle
            is_vehicle_possible, vehicle = is_possible(child_establishments[lower_index:upper_index])
            if(not is_vehicle_possible):
                break
            else:
                upper_index += 1

        if(not is_vehicle_possible):
            child["vehicles"].append(last_possible_vehicle)
            unvisited_establishments = unvisited_establishments.difference(last_possible_vehicle["establishments"])

        else:
            child["vehicles"].append(vehicle)
            unvisited_establishments = unvisited_establishments.difference(vehicle["establishments"])


    if(len(child["vehicles"])<num_vehicles):
        """
        diff = num_vehicles-len(child["vehicles"])
         print(f"{diff} vehicles added")
        print(f"{len(unvisited_establishments)} unvisited establishments") """
        unvisited_establishments_list = list(unvisited_establishments)
        random.shuffle(unvisited_establishments_list)
        lower_index=0
        upper_index=0
        while upper_index < len(unvisited_establishments_list) and len(child["vehicles"])<num_vehicles:
            lower_index = upper_index
            is_vehicle_possible, vehicle = is_possible(unvisited_establishments_list[lower_index:upper_index])
            last_possible_vehicle = vehicle
            while upper_index < len(unvisited_establishments_list):
                last_possible_vehicle = vehicle
                is_vehicle_possible, vehicle = is_possible(unvisited_establishments_list[lower_index:upper_index])
                if(not is_vehicle_possible):
                    break
                else:
                    upper_index += 1

            if(not is_vehicle_possible):
                child["vehicles"].append(last_possible_vehicle)
                unvisited_establishments = unvisited_establishments.difference(last_possible_vehicle["establishments"])

            else:
                child["vehicles"].append(vehicle)
                unvisited_establishments = unvisited_establishments.difference(vehicle["establishments"])
    
    """"
        print(child)
        print()
        print()
        print(f"{len(unvisited_establishments)} unvisited establishments")
        print()
        print() 
    """

    child["unvisited_establishments"] = list(unvisited_establishments)

    return child

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

def populate_vehicle(univisited_establishments):
    vehicle = {"establishments":[],
               "current_time":datetime.time(9, 0),
              }
    for establishment in univisited_establishments:
        end_of_inspection = can_visit(vehicle,establishment)
        if end_of_inspection:
            vehicle["establishments"].append(establishment)
            vehicle["current_time"] = end_of_inspection
        else:
            break
    for establishment in vehicle["establishments"]:
        univisited_establishments.remove(establishment)
    return vehicle, univisited_establishments


def populate_solution(solution_1_establishments):
    solution = {"vehicles":[{"establishments":[],
               "current_time":datetime.time(9, 0),
              } for _ in range(0,num_vehicles)],
              "unvisited_establishments":list(range(1,num_establishments+1))}


    for establishment in solution_1_establishments:
        vehicles_to_check = list(range(0,num_vehicles))
        for vehicle_to_check in vehicles_to_check:
            end_of_inspection = can_visit(solution["vehicles"][vehicle_to_check],establishment)
            if end_of_inspection:
                solution["vehicles"][vehicle_to_check]["establishments"].append(establishment)
                solution["unvisited_establishments"].remove(establishment)
                solution["vehicles"][vehicle_to_check]["current_time"] = end_of_inspection
                break
    for vehicle in solution["vehicles"]:
        if vehicle["establishments"] == []:
            vehicle, solution["unvisited_establishments"] = populate_vehicle(solution["unvisited_establishments"])
        time_to_depot = distances.loc[f'p_{vehicle["establishments"][-1]}']['p_0']
        vehicle["current_time"] = add_seconds(vehicle["current_time"],time_to_depot)

    return solution

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


""" s1 = generate_random_solution()
print(s1)
print()
s2 = generate_random_solution()
print(s2)

print()
print(lox_crossover(s1,s2)) """

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


    best_solution = max(population,key=evaluate_solution) # Initial solution
    best_score = evaluate_solution(best_solution)
    print(f"Initial best solution: {best_solution}")
    print(f"Initial best score: {best_score}")
    best_solution_generation = 0 # Generation on which the best solution was found
    
    generation_no = 0
    
    
    while(num_iterations > 0):
        
        generation_no += 1
        print(f"Generation {generation_no}")
        random_winner_sol = random.choice(population)
        tournment_winner_sol = tournament_select(population,5)
        roulette_winner_sol = roulette_select(population)
        
        """
        children = []
        for i in range(8):
            children.append(crossover_func(random_winner_sol,roulette_winner_sol))
            children.append(crossover_func(random_winner_sol,tournment_winner_sol))
            children.append(crossover_func(roulette_winner_sol,tournment_winner_sol))
        
        
        for child in children:
            if(random.random()<0.03):
                child = mutation_func(child)
            replace_least_fittest(population,child)
        """
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
            if log:
                print(f"\nGeneration: {generation_no }")
                print(f"Solution: {best_solution}, score: {best_score}")
                print_population(population)
        else:
            num_iterations -= 1
        
    print(f"  Final solution score: {best_score}")
    print(f"  Found on generation {best_solution_generation}")
    
    return best_solution

def tabu_search(initial_solution, max_iterations, tabu_list_size):
    current_solution = initial_solution
    best_solution = initial_solution
    tabu_list = []
    iteration = 0
    
    while iteration < max_iterations:
        print(f"Iteration {iteration}")
        iteration += 1
        best_neighbor = None
        neighbor = get_neighbor_solution(current_solution)
        if neighbor not in tabu_list:
            if best_neighbor is None or evaluate_solution(neighbor) > evaluate_solution(best_neighbor):
                best_neighbor = neighbor
        if best_neighbor is None:
            print("No neighbor found")
            break
        current_solution = best_neighbor
        if evaluate_solution(current_solution) > evaluate_solution(best_solution):
            best_solution = current_solution
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
    return best_solution

#print(establishments["Inspection Time"].mean())

#best_solution = genetic_algorithm(100, 50, lox_crossover, mutate_solution)
#print(best_solution) 


print(genetic_algorithm(1000,50,final_crossover,mutate_solution,log=True))

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
print(child_2)  """


#get_sa_solution(1000)

""" 
initial_solution = generate_random_solution()
print(initial_solution)
print(evaluate_solution(initial_solution))
final = tabu_search(initial_solution, 600, 80)
print(final)
print(evaluate_solution(final))  
"""

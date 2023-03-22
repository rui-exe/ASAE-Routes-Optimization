import copy
import random
import utils

def change_two_establishments_in_vehicle(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"]) <= 1:
        return solution
    establishment_1 = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    establishments_to_visit = list(range(0,len(neighbor["vehicles"][vehicle]["establishments"])))
    establishment_2 = random.choice(establishments_to_visit)
    while establishment_2==establishment_1:
        establishment_2 = random.choice(establishments_to_visit)

    neighbor["vehicles"][vehicle]["establishments"][establishment_1],neighbor["vehicles"][vehicle]["establishments"][establishment_2] = neighbor["vehicles"][vehicle]["establishments"][establishment_2],neighbor["vehicles"][vehicle]["establishments"][establishment_1]
    (_,new_vehicle)=utils.is_possible_penalty(neighbor["vehicles"][vehicle]["establishments"])
    neighbor["vehicles"][vehicle]=new_vehicle
    return neighbor
            

def change_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"])>1:
        establishment_1_index = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    elif len(neighbor["vehicles"][vehicle]["establishments"])==1:
        establishment_1_index = 0
    else:
        return neighbor
    establishment_to_mark_as_unvisited =  neighbor["vehicles"][vehicle]["establishments"][establishment_1_index]
    establishments_to_visit = copy.deepcopy(neighbor["unvisited_establishments"])
    establishment_to_visit = random.choice(establishments_to_visit)
    
    neighbor["unvisited_establishments"].remove(establishment_to_visit)
    neighbor["unvisited_establishments"].append(establishment_to_mark_as_unvisited)
    neighbor["vehicles"][vehicle]["establishments"][establishment_1_index] = establishment_to_visit

    (_,new_vehicle)=utils.is_possible_penalty(neighbor["vehicles"][vehicle]["establishments"])
    neighbor["vehicles"][vehicle]=new_vehicle
    return neighbor


def remove_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"])>1:
        establishment_index = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    elif len(neighbor["vehicles"][vehicle]["establishments"])==1:
        establishment_index = 0
    else:
        return neighbor
    establishment_to_mark_as_unvisited =  neighbor["vehicles"][vehicle]["establishments"][establishment_index]
    neighbor["unvisited_establishments"].append(establishment_to_mark_as_unvisited)
    neighbor["vehicles"][vehicle]["establishments"].pop(establishment_index)
    (_,new_vehicle)=utils.is_possible_penalty(neighbor["vehicles"][vehicle]["establishments"])
    neighbor["vehicles"][vehicle] = new_vehicle
    return neighbor
    

def add_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    establishment_to_visit = random.choice(neighbor["unvisited_establishments"])
    
    neighbor["unvisited_establishments"].remove(establishment_to_visit)
    neighbor["vehicles"][vehicle]["establishments"].append(establishment_to_visit)

    (_,new_vehicle)=utils.is_possible_penalty(neighbor["vehicles"][vehicle]["establishments"])
    neighbor["vehicles"][vehicle]=new_vehicle
    return neighbor

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
    best_solution = []
    best_solution_utility = float('-inf')

    for i in range(6):
        neighbor_function = random.choice([add_random_establishment, remove_random_establishment, change_random_establishment, change_two_establishments_in_vehicle])
        new_solution = neighbor_function(solution)
        new_solution_utility,penalty = utils.evaluate_solution_with_penalty(new_solution)
        new_solution_utility -= penalty
        if new_solution_utility > best_solution_utility:
            best_solution_utility = new_solution_utility
            best_solution = new_solution
            
    return best_solution
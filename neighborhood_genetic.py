import random
import copy
import utils
import datetime

"""
    Generates a neighbor solution given a solution
    Does not allow infeasible solutions
"""
"""
    Changes two establishments in a vehicle
    Parameters: solution
    Returns: neighbor solution
"""
def change_two_establishments_in_vehicle(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    if len(neighbor["vehicles"][vehicle]["establishments"]) == 0:
        return solution
    establishment_1 = random.randint(0,len(neighbor["vehicles"][vehicle]["establishments"])-1)
    establishments_to_visit = list(range(0,len(neighbor["vehicles"][vehicle]["establishments"])-1))
    random.shuffle(establishments_to_visit)
    for establishment_2 in establishments_to_visit:
        if establishment_1 != establishment_2:
            neighbor["vehicles"][vehicle]["establishments"][establishment_1],neighbor["vehicles"][vehicle]["establishments"][establishment_2] = neighbor["vehicles"][vehicle]["establishments"][establishment_2],neighbor["vehicles"][vehicle]["establishments"][establishment_1]
            (is_vehicle_possible,new_vehicle)=utils.is_possible(neighbor["vehicles"][vehicle]["establishments"])
            if is_vehicle_possible:
                neighbor["vehicles"][vehicle]=new_vehicle
                return neighbor
            else:
                neighbor["vehicles"][vehicle]["establishments"][establishment_1],neighbor["vehicles"][vehicle]["establishments"][establishment_2] = neighbor["vehicles"][vehicle]["establishments"][establishment_2],neighbor["vehicles"][vehicle]["establishments"][establishment_1] # Undo change
            
    return solution
"""
    Changes two establishments in two different vehicles
    Parameters: solution
    Returns: neighbor solution
"""
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
    random.shuffle(establishments_to_visit)
    
    for establishment_to_visit in establishments_to_visit:
        neighbor["unvisited_establishments"].remove(establishment_to_visit)
        neighbor["unvisited_establishments"].append(establishment_to_mark_as_unvisited)
        neighbor["vehicles"][vehicle]["establishments"][establishment_1_index] = establishment_to_visit

        (is_vehicle_possible,new_vehicle)=utils.is_possible(neighbor["vehicles"][vehicle]["establishments"])
        if is_vehicle_possible:
            neighbor["vehicles"][vehicle]=new_vehicle
            return neighbor
        else:
            neighbor["unvisited_establishments"].append(establishment_to_visit)
            neighbor["unvisited_establishments"].remove(establishment_to_mark_as_unvisited)
            neighbor["vehicles"][vehicle]["establishments"][establishment_1_index] = establishment_to_mark_as_unvisited

    return solution
"""
    Calculates the current time of a vehicle
    Parameters: vehicle
    Returns: current time
"""
def calculate_current_time(vehicle):
    current_time = datetime.time(9, 0)
    establishments = range(1,len(vehicle["establishments"]))
    for establishment in establishments:
        current_time = utils.add_seconds(current_time,utils.distances.loc[f'p_{vehicle["establishments"][establishment-1]}'][f'p_{vehicle["establishments"][establishment]}'])
    return current_time
"""
    Removes a random establishment from a vehicle
    Parameters: solution
    Returns: neighbor solution
"""
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
    neighbor["vehicles"][vehicle]["current_time"] = calculate_current_time(neighbor["vehicles"][vehicle])
    return neighbor
    
"""
    Adds a random establishment to a vehicle
    Parameters: solution
    Returns: neighbor solution
"""
def add_random_establishment(solution):
    neighbor = copy.deepcopy(solution)
    vehicle = random.randint(0,utils.num_vehicles-1)
    establishments_to_visit = copy.deepcopy(neighbor["unvisited_establishments"])
    random.shuffle(establishments_to_visit)
    
    for establishment_to_visit in establishments_to_visit:
        neighbor["unvisited_establishments"].remove(establishment_to_visit)
        neighbor["vehicles"][vehicle]["establishments"].append(establishment_to_visit)

        (is_vehicle_possible,new_vehicle)=utils.is_possible(neighbor["vehicles"][vehicle]["establishments"])
        if is_vehicle_possible:
            neighbor["vehicles"][vehicle]=new_vehicle
            return neighbor
        else:
            neighbor["unvisited_establishments"].append(establishment_to_visit)
            neighbor["vehicles"][vehicle]["establishments"].remove(establishment_to_visit)
    return solution
"""
    Generates a neighbor solution given a solution
    Does not allow infeasible solutions
    Parameters: solution
    Returns: neighbor solution
"""
def get_neighbor_solution(solution):
    neighbor_function = random.choice([add_random_establishment, remove_random_establishment, change_random_establishment, change_two_establishments_in_vehicle])
    return neighbor_function(solution)
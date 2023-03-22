import random
import datetime
import copy
from collections import Counter
from functools import reduce

distances = None
establishments = None
num_establishments = None
num_vehicles = None
END_OF_SHIFT = None

average_time_spent_to_inspect = None

def init_variables(distances_main, establishments_main, num_establishments_main, num_vehicles_main, END_OF_SHIFT_MAIN):
    global distances
    global establishments
    global num_establishments
    global num_vehicles
    global END_OF_SHIFT
    global average_time_spent_to_inspect
    distances = distances_main
    establishments = establishments_main
    num_establishments = num_establishments_main
    num_vehicles = num_vehicles_main
    END_OF_SHIFT = END_OF_SHIFT_MAIN
    average_traveling_between_establishments = (distances.mean().mean())/60
    average_inspection_time = establishments["Inspection Time"].mean()
    average_time_spent_to_inspect = 20




    

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
               "closed_establishments":0
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


def evaluate_solution_with_penalty(solution):
    visited_establishments = num_establishments-len(solution["unvisited_establishments"])
    penalty = 0
    #calculate time 
    for penalty_func in [penalty_repeated_establishments,penalty_establishments_schedule,penalty_overtime_vehicles]:
        penalty+=penalty_func(solution)

    return visited_establishments,penalty


def penalty_repeated_establishments(solution):
    penalty = 0
    all_establishments = reduce(lambda current_establishments,other_vehicle:current_establishments+other_vehicle["establishments"],solution["vehicles"],[])
    counts = Counter(all_establishments)
    unique_establishments = set(all_establishments)
    for establishment in unique_establishments:
        times_repeated = counts[establishment]-1
        if(times_repeated>0):
            penalty+=times_repeated
    return penalty

def overtime(time):
    datetime_reach_depot = datetime.datetime.combine(datetime.datetime.today(), time)
    datetime_1700 = datetime.datetime.combine(datetime.datetime.today(), END_OF_SHIFT)
    time_diff = datetime_reach_depot - datetime_1700
    return time_diff.total_seconds()/60


def can_visit_penalty(vehicle,establishment):
    if(len(vehicle["establishments"])==0):
        time_to_establishment = distances.loc['p_0'][f'p_{establishment}']
    else:
        current_establishment = vehicle["establishments"][-1]
        time_to_establishment = distances.loc[f'p_{current_establishment}'][f'p_{establishment}']

    establishment_opening_hours = establishments.iloc[establishment]["Opening Hours"] # Get list with the working hours of the establishment
    inspection_duration = establishments.iloc[establishment]["Inspection Time"].item()
    current_time = vehicle["current_time"]


    arriving_time = add_seconds(current_time,time_to_establishment) # Add distance to current time
    inspection_start = copy.deepcopy(arriving_time)
    while(not establishment_opening_hours[inspection_start.hour]):
        if(inspection_start.hour+1>=17):
            return 1,arriving_time
        inspection_start = datetime.time(inspection_start.hour+1, 0)


    end_of_inpection = add_minutes(inspection_start,inspection_duration) #Add inspection time to arriving time

    return 0,end_of_inpection

def is_possible_penalty(establishments):
    vehicle={"establishments":[],
               "current_time":datetime.time(9, 0),
              } 
    
    penalty=0

    for establishment in establishments:
        local_penalty,end_of_inspection = can_visit_penalty(vehicle,establishment)
        vehicle["establishments"].append(establishment)
        vehicle["current_time"] = end_of_inspection
        penalty+=local_penalty 


    if vehicle["establishments"]:
        time_to_depot = distances.loc[f'p_{vehicle["establishments"][-1]}']['p_0']
        vehicle["current_time"] = add_seconds(vehicle["current_time"],time_to_depot)
    
    vehicle["closed_establishments"] = penalty
    
    return penalty,vehicle

def penalty_establishments_schedule(solution):
    return sum(map(lambda vehicle:vehicle["closed_establishments"],solution["vehicles"]))

def penalty_overtime_vehicles(solution):
    penalty = 0 
    for vehicle in solution["vehicles"]:
        extra_minutes = overtime(vehicle["current_time"])
        if(extra_minutes>0):
            penalty+=(extra_minutes/average_time_spent_to_inspect)
    return penalty

def is_legal(solution):
    return all(map(lambda vehicle:is_possible(vehicle["establishments"]),solution["vehicles"])) and evaluate_solution(solution) == num_establishments - len(solution["unvisited_establishments"])
import utils
import numpy as np
import random
import copy
import math
from neighborhood_with_unfeasible import get_neighbor_solution
import time
import datetime
import matplotlib.pyplot as plt


INITIAL_STAGE = 0
INITIAL_STAGE_ITERATIONS = 1

IMTERMEDIATE_STAGE_TEMPERATURE = 1000
INTERMEDIATE_STAGE = 0
INTERMEDIATE_STAGE_ITERATIONS = 10

FINAL_STAGE_TEMPERATURE = 10
FINAL_STAGE = 0
FINAL_STAGE_ITERATIONS = 30

ITERATIONS_PER_STAGE = [INITIAL_STAGE_ITERATIONS,INTERMEDIATE_STAGE_ITERATIONS,FINAL_STAGE_ITERATIONS]


def get_sa_solution(num_iterations, log=False):
    iteration = 0
    temperature = 1000
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score
    nr_establishments = num_iterations/25
    COOLING_RATE = 1-(1/(nr_establishments))
    while iteration < num_iterations:
        temperature = temperature * COOLING_RATE  
        iteration += 1

        neighbor = get_neighbor_solution(solution)

        neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
        penalty_factor = math.log(1/temperature)
        if penalty_factor<1:
            penalty_factor=1
        neighbor_score_with_penalty = neighbor_score_without_penalty - penalty_factor*penalty

        delta_e = neighbor_score_with_penalty-score

        if delta_e>0 or np.exp(delta_e/temperature)>random.random():
            
            score = neighbor_score_with_penalty
            solution = neighbor 
            print(score , temperature)


            if(neighbor_score_without_penalty>best_score and penalty==0):
                iteration=1
                best_score=neighbor_score_without_penalty
                best_solution=solution
                print("New best score: ",best_score)

            
    print("Final solution: ",best_solution)
    print("Final score: ",best_score)
    return best_solution 



def get_sa_solution_adaptive_with_stages(num_iterations, log=False):
    start = time.time()
    iteration = 0
    temperature = 1000000
    threshold = 1000
    iterations_since_improvement = 0
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score
    iterations_at_temp_scaling_factor = 1
    nr_establishments = num_iterations/25
    COOLING_RATE = 1-(1/(nr_establishments))

    while iteration < num_iterations:
        temperature = temperature * COOLING_RATE  
        iteration += 1

        if temperature>IMTERMEDIATE_STAGE_TEMPERATURE:
            current_stage = INITIAL_STAGE
        elif temperature>FINAL_STAGE_TEMPERATURE:
            current_stage = INTERMEDIATE_STAGE
        else:
            current_stage = FINAL_STAGE

        iterations_at_temp = ITERATIONS_PER_STAGE[current_stage]
        i=0
        while i<iterations_at_temp:
            neighbor = get_neighbor_solution(solution)

            neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
            penalty_factor = math.log(1/temperature)
            if penalty_factor<1:
                penalty_factor=1
            neighbor_score_with_penalty = neighbor_score_without_penalty - penalty_factor*penalty

            delta_e = neighbor_score_with_penalty-score

            if delta_e>0 or np.exp(delta_e/temperature)>random.random():
                
                score = neighbor_score_with_penalty
                solution = neighbor 

                if(neighbor_score_without_penalty>best_score and penalty==0):
                    iteration=1
                    iterations_at_temp_scaling_factor = 1
                    iterations_at_temp = iterations_at_temp_scaling_factor*ITERATIONS_PER_STAGE[current_stage]
                    best_score=neighbor_score_without_penalty
                    best_solution=solution
            else:
                iterations_since_improvement += 1

            if iterations_since_improvement >= threshold:
                iterations_at_temp_scaling_factor+=1
                iterations_at_temp = iterations_at_temp_scaling_factor*ITERATIONS_PER_STAGE[current_stage]
                iterations_since_improvement = 0
            

            i+=1

    return best_solution 
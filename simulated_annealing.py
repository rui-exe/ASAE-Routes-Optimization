import utils
import numpy as np
import random
import copy
import math
from neighborhood_with_unfeasible import get_neighbor_solution
import time


def get_sa_solution(num_iterations, log=False):
    start = time.time()

    iteration = 0
    temperature = 1000000
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score
    print(solution,best_score)

    while iteration < num_iterations:
        temperature = temperature * 0.99  # Test with different cooling schedules
        iteration += 1
        neighbor = get_neighbor_solution(solution)

        neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
        neighbor_score_with_penalty = neighbor_score_without_penalty - penalty

        delta_e = neighbor_score_with_penalty-score

        #print(f"Temperature: {temperature}, delta_e:{delta_e}")
        if delta_e>0 or np.exp(delta_e/temperature)>random.random():
            
            score = neighbor_score_with_penalty
            solution = neighbor 
            
            #if(delta_e<0):
                #print(f"Temperature {temperature}, probability: {np.exp(delta_e/temperature)} ")

            if(neighbor_score_without_penalty>best_score and penalty==0):
                end = time.time()
                print(f"Elapsed time{end - start}")
                print(f"Current score: {neighbor_score_without_penalty} found at {iteration}")
                print(solution)
                iteration=1
                best_score=neighbor_score_without_penalty
                best_solution=solution

        if log:
            print(f"Solution: {best_solution}, score: {best_score},  Temp: {temperature}")

    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution 

""" def generate_t0(current_solution,n_attempts,alfa):

    # define the number of attempts
    current_solution_score, penalty = utils.evaluate_solution_with_penalty(current_solution)
    current_solution_score -= penalty

    # calculate the average absolute difference in the objective function value between a solution and its neighboring solutions
    delta_c_values = []

    for i in range(n_attempts):
        print(f"Attempt {i}")
        # generate a neighboring solution
        neighbor = get_neighbor_solution(current_solution)

        neighbor_score, penalty = utils.evaluate_solution_with_penalty(neighbor)
        neighbor_score -= penalty

        # calculate the absolute difference in the objective function value between the current solution and the neighboring solution
        delta_c_values.append(abs(neighbor_score - current_solution_score))
    
    delta_c_values = np.array(delta_c_values)
    delta_c_average = np.mean(delta_c_values)
    delta_c_stddev = np.mean(delta_c_values)



    return (delta_c_average + 3 * delta_c_stddev) / math.log(1 / alfa)

REHEAT_MAX = 5
MARKOV_CHAIN_WITHOUT_IMPROVEMENT_TRESHOLD = 500
K_MAX = 30
def simulated_annealing(tf, stable_temperature, max_iter, a=2, f=1/3):
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score

    t0 = 100000000
    temp = t0
    reheat=0
    while reheat < REHEAT_MAX: #Reheat
        current_solution = solution
        i = 0
        score,_ = utils.evaluate_solution_with_penalty(solution)
        markov_chain_number = 0
        chains_without_update = 0 #Markov chains without update
        while ((i < max_iter or chains_without_update < MARKOV_CHAIN_WITHOUT_IMPROVEMENT_TRESHOLD) and markov_chain_number < K_MAX): #Stopping criteria

            neighbor = get_neighbor_solution(solution)

            neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
            neighbor_score_with_penalty = neighbor_score_without_penalty - penalty


            utility_diff = neighbor_score_with_penalty-score
            prob = min (utility_diff/temp, 700)
            if (random.random() <= math.exp(prob)): #acceptance condition
                current_solution = neighbor
                score = neighbor_score_with_penalty
            if i >= max_iter:
                if utility_diff > 0:
                    chains_without_update = 0
                else:
                    chains_without_update += 1
            markov_chain_number += 1
            #########Cooling schedule#############
            P = math.log(math.log(t0/tf)/math.log(a))
            Q = math.log(1/f)
            b = P/Q
            alpha = math.exp(-1/((i/t0)**b))
            if i > stable_temperature:
                temp = t0 * alpha
                print(temp)
            elif i > tf:
                temp = 0
            if score > best_score and penalty==0: #Updates the best solution
                print(f"New best score:{best_score}")
                best_score = score
                best_solution = current_solution
            i += 1
        reheat += 1
        temp = t0
    return best_solution """
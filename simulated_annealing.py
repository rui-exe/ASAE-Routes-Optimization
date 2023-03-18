import utils
import numpy as np
import random
import copy
import math
from neighborhood_with_unfeasible import get_neighbor_solution

""" def get_sa_solution(num_iterations, log=False):
    iteration = 0
    temperature = 1000000
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score
    print(solution,best_score)

    while iteration < num_iterations:
        temperature = temperature * 0.999  # Test with different cooling schedules
        iteration += 1
        neighbor = get_neighbor_solution(solution)

        neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
        neighbor_score_with_penalty = neighbor_score_without_penalty - penalty

        delta_e = neighbor_score_with_penalty-score

        if delta_e>0 or np.exp(delta_e/temperature)>random.random():
            score = neighbor_score_with_penalty
            solution = neighbor 

            if(neighbor_score_without_penalty>best_score and penalty==0):
                print(f"Current score: {neighbor_score_without_penalty}")
                iteration=1
                best_score=neighbor_score_without_penalty
                best_solution=solution

        if log:
            print(f"Solution: {best_solution}, score: {best_score},  Temp: {temperature}")

    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution """

def generate_t0(current_solution,n_attampts,alfa):

    # define the number of attempts
    current_solution_score, penalty = utils.evaluate_solution_with_penalty(current_solution)
    current_solution_score -= penalty

    # calculate the average absolute difference in the objective function value between a solution and its neighboring solutions
    delta_c_values = []

    for i in range(n_attampts):
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

def simulated_annealing(tf, max_iter, a=2, f=1/3):
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score

    t0 = generate_t0(solution,1000,0.9)
    k = 0
    t0 = (delta_c + 3 * sigma_delta_c) / math.log(1 / alpha_0)

    while k < max_iter and t > tf:
        # Generate a new candidate state
        neighbor = get_neighbor_solution(solution)

        neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
        neighbor_score_with_penalty = neighbor_score_without_penalty - penalty

        # Determine whether to accept the candidate state
        delta_energy = neighbor_score_with_penalty - score
        if delta_energy>0 or np.exp(delta_energy/t)>random.random():
            score = neighbor_score_with_penalty
            solution = neighbor 

            if(neighbor_score_without_penalty>best_score and penalty==0):
                print(f"Current score: {neighbor_score_without_penalty}")
                best_score=neighbor_score_without_penalty
                best_solution=solution
        # Update the temperature
        k += 1
        P = math.log(math.log(t0/tf)/math.log(a))
        Q = math.log(1/f)
        b = P/Q
        alpha = math.exp(-1/((k/t0)**b))
        t = t0 * alpha
    return best_solution
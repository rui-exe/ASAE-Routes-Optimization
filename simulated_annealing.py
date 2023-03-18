import utils
import numpy as np
import random
import copy
from neighborhood_with_unfeasible import get_neighbor_solution

def get_sa_solution(num_iterations, log=False):
    iteration = 0
    temperature = 1000000
    solution = utils.generate_random_solution()
    score,_ = utils.evaluate_solution_with_penalty(solution)
    best_solution = copy.deepcopy(solution)
    best_score = score

    print(f"Init Solution:  {best_solution}, score: {best_score}")

    while iteration < num_iterations:
        temperature = temperature * 0.999  # Test with different cooling schedules
        iteration += 1
        print(f"Iteration {iteration}")
        neighbor = get_neighbor_solution(solution)

        neighbor_score_without_penalty,penalty = utils.evaluate_solution_with_penalty(neighbor)
        neighbor_score_with_penalty = neighbor_score_without_penalty - penalty

        delta_e = neighbor_score_with_penalty-score

        if delta_e>0 or np.exp(delta_e/temperature)>random.random():
            score = neighbor_score_with_penalty
            solution = neighbor 

            if(neighbor_score_without_penalty>best_score):
                iteration=1
                best_score=score
                best_solution=solution

        if log:
            print(f"Solution: {best_solution}, score: {best_score},  Temp: {temperature}")

    print(f"Final Solution: {best_solution}, score: {best_score}")
    return best_solution

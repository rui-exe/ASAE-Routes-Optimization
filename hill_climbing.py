import utils
from neighborhood_with_unfeasible import get_neighbor_solution
import time
import matplotlib.pyplot as plt
import datetime



"""
    Hill Climbing Algorithm
    Parameters:
        num_iterations: number of iterations to run the algorithm
        log: if True, prints the solution and score at each iteration
    Returns:
        final_solution: the best solution found
"""

def get_hc_solution(num_iterations, log=False):
    iteration = 0
    best_solution = utils.generate_random_solution()
    final_solution = best_solution
    best_score,_ = utils.evaluate_solution_with_penalty(best_solution)
    print("Initial solution: ",best_solution)
    print("Initial solution score: ",best_score)
    establishments_visited = utils.num_establishments-len(best_solution["unvisited_establishments"])
    start_time = time.time()
    solution_utilities = [best_score]
    times = [0]
    while iteration < num_iterations:
        #print(f"Iteration {iteration}")
        iteration += 1
        neighbor = get_neighbor_solution(best_solution)
        solution_utility,penalty = utils.evaluate_solution_with_penalty(neighbor)
        solution_utility -= penalty

        if solution_utility > best_score:
            best_score = solution_utility
            best_solution = neighbor
            if(penalty==0):
                print("New best score: ",best_score)
                final_solution=best_solution
                solution_utilities.append(solution_utility)
                times.append(time.time()-start_time)
            iteration=0 

        if log:
            (print(f"Solution:       {best_solution}, score: {best_score}"))

    plt.plot(times, solution_utilities)
    plt.title('Solution Utility over Time (HC) -  ' + str(utils.num_establishments) + ' Establishments')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Solution Utility')
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig('plots/hc_solution_utility' + str(date) + '.png')
    print("Final solution: ",final_solution)
    return final_solution
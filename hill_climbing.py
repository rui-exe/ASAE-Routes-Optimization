import utils
from neighborhood_with_unfeasible import get_neighbor_solution

def get_hc_solution(num_iterations, log=False):
    iteration = 0
    best_solution = utils.generate_random_solution()
    final_solution = best_solution
    best_score,_ = utils.evaluate_solution_with_penalty(best_solution)
    print(best_score)
    establishments_visited = utils.num_establishments-len(best_solution["unvisited_establishments"])

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
                print(solution_utility)
                final_solution=best_solution
            iteration=0 

        if log:
            (print(f"Solution:       {best_solution}, score: {best_score}"))

    return final_solution
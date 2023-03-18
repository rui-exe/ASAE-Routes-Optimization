import utils
from neighborhood_with_unfeasible import get_neighbor_solution

def get_hc_solution(num_iterations, log=False):
    iteration = 0
    best_solution = utils.generate_random_solution()
    best_score = utils.evaluate_solution_with_penalty(best_solution,1)
    establishments_visited = utils.num_establishments-len(best_solution["unvisited_establishments"])

    while iteration < num_iterations:
        print(f"Iteration {iteration}")
        iteration += 1
        neighbor = get_neighbor_solution(best_solution)
        solution_utility = utils.evaluate_solution_with_penalty(neighbor,iteration)

        if solution_utility > best_score:
            best_score = solution_utility
            best_solution = neighbor
            iteration=0 
            print(best_solution)
            visited_establishments = utils.evaluate_solution_with_penalty(neighbor,iteration)
            print(visited_establishments)

        if log:
            (print(f"Solution:       {best_solution}, score: {best_score}"))

    print(f"Random Solution Score: {establishments_visited}")
    establishments_visited = utils.num_establishments-len(best_solution["unvisited_establishments"])
    print(f"Final Solution Score: {establishments_visited}")
    return best_solution
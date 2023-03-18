from neighborhood_with_unfeasible import get_neighbor_solution
import utils

def tabu_search(max_iterations, tabu_list_size):
    current_solution = utils.generate_random_solution() 
    best_solution = current_solution
    best_value = utils.evaluate_solution_with_penalty(best_solution,1)
    print(f"Initial value:{best_value}")
    tabu_list = []
    iteration = 0
    
    while iteration < max_iterations:
        print(f"Iteration {iteration}")
        iteration += 1
        best_neighbor = None
        neighbor = get_neighbor_solution(current_solution)
        if neighbor not in tabu_list:
            if best_neighbor is None or utils.evaluate_solution_with_penalty(neighbor,iteration) > utils.evaluate_solution_with_penalty(best_neighbor,iteration):
                best_neighbor = neighbor
        if best_neighbor is None:
            print("No neighbor found")
            break
        current_solution = best_neighbor
        current_value = utils.evaluate_solution_with_penalty(current_solution,iteration)
        if current_value > best_value:
            best_solution = current_solution
            best_value = current_value
        tabu_list.append(best_neighbor)
        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)
    return best_solution
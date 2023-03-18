import datetime
import pandas as pd
import ast
import math
from utils import init_variables,is_legal,evaluate_solution
from simulated_annealing import simulated_annealing
from hill_climbing import get_hc_solution

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


num_establishments = 200

num_vehicles = math.floor(0.1*num_establishments)

END_OF_SHIFT = datetime.time(17, 0)

init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)

init_time = datetime.datetime.now()

solution = simulated_annealing(0.001,0.1,1000)
print(is_legal(solution))
print(solution)
print(evaluate_solution(solution))


final_time = datetime.datetime.now()
print(final_time-init_time)
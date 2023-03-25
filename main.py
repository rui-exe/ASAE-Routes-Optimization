import datetime
import pandas as pd
import ast
import math
from utils import init_variables,is_legal,evaluate_solution
from simulated_annealing import get_sa_solution,get_sa_solution_adaptive_with_stages
from hill_climbing import get_hc_solution
from multiprocessing import Process

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


num_establishments = 100

num_vehicles = math.floor(0.1*num_establishments)

END_OF_SHIFT = datetime.time(17, 0)

init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)

init_time = datetime.datetime.now()

def normal_sa():
    solution = get_sa_solution(5000)
    print(f"Normal SA: {evaluate_solution(solution)}")


def adaptive_sa():
    solution = get_sa_solution_adaptive_with_stages(5000)
    print(f"Adapative SA: {evaluate_solution(solution)}")


for i in range(2):
    p1 = Process(target=normal_sa)
    p1.start()

for i in range(2):
    p2 = Process(target=adaptive_sa)
    p2.start()


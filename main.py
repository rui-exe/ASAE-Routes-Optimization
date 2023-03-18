import datetime
import pandas as pd
import ast
import math
from utils import init_variables
from simulated_annealing import get_sa_solution

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


num_establishments = 30

num_vehicles = math.floor(0.1*num_establishments)

END_OF_SHIFT = datetime.time(17, 0)

init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)

init_time = datetime.datetime.now()

print(get_sa_solution(200))


final_time = datetime.datetime.now()
print(final_time-init_time)
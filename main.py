import datetime
import pandas as pd
import ast
import math
from utils import init_variables
from genetic import *


distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


#num_establishments = len(establishments) - 1
num_establishments = 30

num_vehicles = math.floor(0.1*num_establishments)

END_OF_SHIFT = datetime.time(17, 0)

init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)

print(genetic_algorithm(1000,50,final_crossover,mutate_solution,log=True))
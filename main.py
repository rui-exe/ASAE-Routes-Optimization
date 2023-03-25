import datetime
import pandas as pd
import ast
import math
from utils import init_variables,is_legal,evaluate_solution
from simulated_annealing import simulated_annealing
from hill_climbing import get_hc_solution
from genetic import genetic_algorithm, final_crossover, mutate_solution
import tkinter as tk
from gmplot import gmplot
import webbrowser
import os 
import random

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)

END_OF_SHIFT = datetime.time(17, 0)


class EstablishmentGUI:
    def __init__(self, master):
        self.master = master
        master.title("ASAE - Vehicle Routing Problem")
        master.protocol("WM_DELETE_WINDOW", self.on_close)

        self.label_est = tk.Label(master, text="Enter the number of establishments:")
        self.label_est.pack()

        self.entry_est = tk.Entry(master)
        self.entry_est.pack()

        self.label_func = tk.Label(master, text="Select the function you want to run:")
        self.label_func.pack()

        self.func_var = tk.StringVar(master)
        self.func_var.set("Hill Climbing")  # default value
        self.func_menu = tk.OptionMenu(master, self.func_var, "Hill Climbing", "Simulated Annealing", "Genetic Algorithm")
        self.func_menu.pack()

    
        self.submit_button = tk.Button(master, text="Submit", command=self.submit)
        self.submit_button.pack()

    def submit(self):
        num_establishments = int(self.entry_est.get())
        num_vehicles = math.floor(0.1* num_establishments)
        function = self.func_var.get()
        
    
        init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)
        if function == "Hill Climbing":
            solution = get_hc_solution(1000)
        elif function == "Simulated Annealing":
            solution = simulated_annealing(5000)
        elif function == "Genetic Algorithm":
            solution = genetic_algorithm(500,50,final_crossover,mutate_solution)

        
        # Prompt the user to select a vehicle
        veh_options = ["Vehicle {}".format(i) for i in range(1, num_vehicles+1)]
        vehicle_window = tk.Toplevel(self.master)
        vehicle_window.protocol("WM_DELETE_WINDOW", self.on_close)
        vehicle_label = tk.Label(vehicle_window, text="Select a vehicle:")
        vehicle_label.pack()
        vehicle_option = tk.StringVar(vehicle_window)
        vehicle_option.set(veh_options[0])
        vehicle_menu = tk.OptionMenu(vehicle_window, vehicle_option, *veh_options)
        vehicle_menu.pack()
        vehicle_button = tk.Button(vehicle_window, text="OK", command=lambda: self.update_map(solution, vehicle_option.get()))
        vehicle_button.pack()
        vehicle_window.wait_window(vehicle_window)

        self.master.destroy()

    def generate_link(self, solution, vehicle):
        gmap = gmplot.GoogleMapPlotter(41.160304, -8.602478, 15)
        # Extract the coordinates of the establishments for the selected vehicle
        draw_establishments = solution['vehicles'][vehicle-1]['establishments']
        establishments_coords = [(41.160304, -8.602478)]+[(establishments.iloc[establishment]['Latitude'], establishments.iloc[establishment]['Longitude']) for establishment in draw_establishments]+[(41.160304, -8.602478)]
        
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        gmap.plot([establishment[0] for establishment in establishments_coords], [establishment[1] for establishment in establishments_coords], color, edge_width=2.5)
    
        # Loop through each establishment and add a marker to the map
        for establishment in establishments_coords:
            # Extract the latitude and longitude of the establishment from its index
            latitude, longitude = establishment
            # Add a marker to the map with the color of the current vehicle
            gmap.marker(latitude, longitude)
        
        gmap.draw("mymap.html")
    
    def update_map(self, solution, vehicle):
        vehicle_num = int(vehicle.split()[1])
        self.generate_link(solution, vehicle_num)
        self.display_link()

    def display_link(self):
        file_path = os.path.join(os.getcwd(), "mymap.html")
        webbrowser.open("file://" + file_path)

    def on_close(self):
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = EstablishmentGUI(root)
    root.mainloop()


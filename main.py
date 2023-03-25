import datetime
import pandas as pd
import ast
import math
from utils import init_variables,is_legal,evaluate_solution
from simulated_annealing import get_sa_solution,get_sa_solution_adaptive_with_stages
from hill_climbing import get_hc_solution
from genetic import genetic_algorithm, final_crossover, mutate_solution
import tkinter as tk
import webbrowser
import os 
import folium
from folium.plugins import MarkerCluster
from folium.plugins import AntPath
import sys

distancesFileName = "distances.csv"
establishmentsFileName = "establishments.csv"


conv = {'Opening Hours': lambda x: ast.literal_eval(x)}

distances = pd.read_csv(distancesFileName,index_col=0)
establishments = pd.read_csv(establishmentsFileName, converters=conv)


num_establishments = 100

num_vehicles = math.floor(0.1*num_establishments)

END_OF_SHIFT = datetime.time(17, 0)

init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)


class EstablishmentGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("ASAE - Vehicle Routing Problem")
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.geometry("400x400")
        self.master.eval('tk::PlaceWindow . center')
        self.master.resizable(False, False)
        self.master.configure(bg="white")

        # adjust label and entry font size
        self.label_est = tk.Label(master, text="Enter the number of establishments:", font=("Arial", 12))
        self.label_est.pack(pady=10)
        self.entry_est = tk.Entry(master, font=("Arial", 12))
        self.entry_est.pack(pady=10)

        # adjust menu and button padding
        self.label_func = tk.Label(master, text="Select the function you want to run:", font=("Arial", 12))
        self.label_func.pack(pady=10)
        self.func_var = tk.StringVar(master)
        self.func_var.set("Hill Climbing")  # default value
        self.func_menu = tk.OptionMenu(master, self.func_var, "Hill Climbing", "Simulated Annealing", "Genetic Algorithm")
        self.func_menu.pack(pady=10) # increase the padding
        self.submit_button = tk.Button(master, text="Submit", command=self.submit, font=("Arial", 12))
        self.submit_button.pack(pady=20) # increase the padding

    def submit(self):
        if not(self.entry_est.get().isnumeric()):
            print("Number of establishments must be an integer")
            self.master.destroy()
            return
        num_establishments = int(self.entry_est.get())
        if num_establishments > len(establishments):
            print("Number of establishments must be less than or equal to {}".format(len(establishments)-1))
            self.master.destroy()
            return
        num_vehicles = math.floor(0.1* num_establishments)
        function = self.func_var.get()
        
        # Run the selected function
        init_variables(distances, establishments, num_establishments, num_vehicles, END_OF_SHIFT)
        if function == "Hill Climbing":
            l = self.loading_window("Hill Climbing")
            solution = get_hc_solution(1000)
        elif function == "Simulated Annealing":
            l = self.loading_window("Simulated Annealing")
            solution = get_sa_solution(25*num_establishments)
        elif function == "Genetic Algorithm":
            l = self.loading_window("Genetic Algorithm")
            solution = genetic_algorithm(1000,100,final_crossover,mutate_solution,True)
        # Destroy the loading window and re-enable the submit button
        l.destroy()
        self.submit_button.config(state="normal")
        
        # Prompt the user to select a vehicle
        veh_options = ["Vehicle {}".format(i) for i in range(1, num_vehicles+1)]
        vehicle_window = tk.Toplevel(self.master)
        vehicle_window.protocol("WM_DELETE_WINDOW", self.on_close)
        vehicle_window.title("Select a vehicle")
        vehicle_window.geometry("400x400")
        self.master.eval(f'tk::PlaceWindow {str(vehicle_window)} center')
        vehicle_window.resizable(False, False)
        vehicle_window.configure(bg="white")
        solution_label = tk.Label(vehicle_window, text="Solution evaluation:", font=("Arial", 12))
        solution_label.pack(pady=10)
        solution_label = tk.Label(vehicle_window, text="{}".format(evaluate_solution(solution)), font=("Arial", 12))
        solution_label.pack(pady=10)
        vehicle_label = tk.Label(vehicle_window, text="Select a vehicle:", font=("Arial", 12))
        vehicle_label.pack(pady=10)
        vehicle_option = tk.StringVar(vehicle_window)
        vehicle_option.set(veh_options[0])
        vehicle_menu = tk.OptionMenu(vehicle_window, vehicle_option, *veh_options)
        vehicle_menu.pack(pady=10)
        vehicle_button = tk.Button(vehicle_window, text="OK", command=lambda: self.update_map(solution, vehicle_option.get()))
        vehicle_button.pack(pady=10)
        vehicle_window.wait_window(vehicle_window)

    def generate_link(self, solution, vehicle):
        # Create the map
        m = folium.Map(location=[41.160304, -8.602478], zoom_start=15)

        # Extract the coordinates of the establishments for the selected vehicle
        draw_establishments = solution['vehicles'][vehicle-1]['establishments']
    
        depot_coords = (41.160304, -8.602478)
        establishments_coords = [depot_coords] + [(establishments.iloc[establishment]['Latitude'], establishments.iloc[establishment]['Longitude']) for establishment in draw_establishments] + [depot_coords]
        # Create a marker for the depot
        folium.Marker(location=depot_coords, icon=folium.Icon(color='green', icon='home')).add_to(m)
        # Create a marker cluster layer
        marker_cluster = MarkerCluster().add_to(m)
        color = 'darkblue'
        # Loop through each establishment and add a marker to the marker cluster layer
        for i, establishment in enumerate(establishments_coords):
            # Extract the latitude and longitude of the establishment from its index
            latitude, longitude = establishment
            if (latitude, longitude) == depot_coords:
                continue
            # Add a marker to the marker cluster layer with the establishment index as its label
            folium.Marker(location=[latitude, longitude],icon=folium.features.DivIcon(html=f'<div style="background-color: #0077c2; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 16px; font-weight: bold">{i}</div>')).add_to(marker_cluster)
        # Create a list of location coordinates
        locations = [(establishment[0], establishment[1]) for establishment in establishments_coords]

        # Add an AntPath to the map with the color of the current vehicle
        AntPath(locations, color=color, dash_array=[10, 20], delay=1000, weight=2.5).add_to(m)

        # Save the map as an HTML file
        m.save("mymap.html")
    
    def update_map(self, solution, vehicle):
        vehicle_num = int(vehicle.split()[1])
        self.generate_link(solution, vehicle_num)
        self.display_link()

    def display_link(self):
        file_path = os.path.join(os.getcwd(), "mymap.html")
        webbrowser.open("file://" + file_path)

    def on_close(self):
        self.master.destroy()
        sys.exit("Program closed by user")
    def loading_window(self, function):
        loading_window = tk.Toplevel(self.master)
        loading_window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.master.eval(f'tk::PlaceWindow {str(loading_window)} center')
        loading_window.title("Loading")
        loading_label = tk.Label(loading_window, text="Loading {}...".format(function), font=("Arial", 12))
        loading_label.pack(pady=10)
        self.submit_button.config(state="disabled")
        self.master.update()
        return loading_window

if __name__ == "__main__":
    root = tk.Tk()
    my_gui = EstablishmentGUI(root)
    root.mainloop()


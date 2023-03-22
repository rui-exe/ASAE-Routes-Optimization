from gmplot import gmplot
from main import *

# Create a map centered on a specific location
gmap = gmplot.GoogleMapPlotter(41.160304, -8.602478, 15)


# Define the coordinates of the points to be plotted
latitude_list = [41.160304, 41.21827]
longitude_list = [-8.602478, -8.54907]

# Add the points to the map using the scatter method
for i in range(len(latitude_list)):
    gmap.marker(latitude_list[i], longitude_list[i])

gmap.plot(latitude_list, longitude_list, 
           'cornflowerblue', edge_width = 2.5)

""" 
lat1, lon1 = 41.160304, -8.602478
lat2, lon2 = 41.21827, -8.54907


# Draw a line between the two points
gmap.plot([lat1, lat2], [lon1, lon2], 'cornflowerblue', edge_width=2.5)
gmap.plot(lat1,lon1)
gmap.plot(lat2,lon2)
 """
# Draw the map
gmap.draw("mymap.html")

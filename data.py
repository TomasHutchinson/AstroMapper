import csv
import numpy as np

# https://exoplanetarchive.ipac.caltech.edu/
# https://exoplanetarchive.ipac.caltech.edu/docs/data.html

filename = "PS_2025.11.02_11.27.45.csv"  # File name

fields = []  # Column names
rows = []    # Data rows

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)  # Reader object

    fields = next(csvreader)  # Read header
    for row in csvreader:     # Read rows
        rows.append(row)

print(fields)

def convert_to_3d_coordinates(row):
    # Extract RA, Dec, and distance values from the row
    try:
        ra = float(row[fields.index('ra')])  # Right Ascension in degrees
        dec = float(row[fields.index('dec')])  # Declination in degrees
        distance = float(row[fields.index('sy_dist')])  # Distance in parsecs
    except:
        return (0,0,0)
    
    # Convert RA and Dec from degrees to radians
    ra_rad = np.radians(ra)
    dec_rad = np.radians(dec)
    
    # Convert to 3D Cartesian coordinates (X, Y, Z)
    X = distance * np.cos(dec_rad) * np.cos(ra_rad)
    Y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    Z = distance * np.sin(dec_rad)
    
    return (X, Y, Z)

# print([convert_to_3d_coordinates(i) for i in rows[5:10]])
# print(len(rows))
#points = [convert_to_3d_coordinates(i) for i in rows[1:]]

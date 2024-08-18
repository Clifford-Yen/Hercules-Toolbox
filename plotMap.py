import pygmt
import os
import argparse
import pandas as pd
import utm

def getHerculesInputData(inputFilePath: str) -> dict:
    ''' getInputData returns a dict with all Hercules'
    parameters as the dict keys and values as the dict values. '''
    with open(inputFilePath) as f:
        lines = f.readlines()
        lines = [line.rstrip().split('=') for line in lines if (line[0] != '#' and line.rstrip() != '')]
    inputData = {}
    heldKey = None
    for line in lines:
        if len(line) == 2 and line[1] != '':
            inputData[line[0].rstrip()] = line[1].lstrip()
            heldKey = None
        elif len(line) == 2 and line[1] == '':
            heldKey = line[0].rstrip()
            inputData[line[0].rstrip()] = []
        elif len(line) == 1:
            inputData[heldKey].append(line[0].split())
        else:
            raise ValueError('Unexpected input file format.')
    return inputData

def plotMap(mapFile='map.png', herculesInputFile='inputfiles/parameters.in',
        overwriteExisting: bool|None = None, includeTopography=False, useColorMap=False, 
        includeStations=False, includeFrame=False, stationNames: list[str]|None = None,
        labelFont='6p,black'):
    if overwriteExisting is None and os.path.exists(mapFile):
        # If the map file exists, ask the user if they want to overwrite it
        overwrite = input(f'The file {mapFile} already exists. Do you want to overwrite it? (y/N) ')
        if overwrite.lower() != 'y':
            return
    elif overwriteExisting is False and os.path.exists(mapFile):
        print(f'The file {mapFile} already exists. Skipping...')
        return
    # Read the domain region from the Hercules input file
    inputData = getHerculesInputData(herculesInputFile)
    origin = [float(inputData[x]) for x in ['region_origin_latitude_deg', 'region_origin_longitude_deg']]
    length_east = float(inputData['region_length_east_m'])
    length_north = float(inputData['region_length_north_m'])
    origin_utm_easting, origin_utm_northing, zone_number, zone_letter = utm.from_latlon(origin[0], origin[1])
    top_right_vertex = utm.to_latlon(origin_utm_easting+length_east, origin_utm_northing+length_north, zone_number, zone_letter)
    region = [origin[1], top_right_vertex[1], origin[0], top_right_vertex[0]]
    # Plot the map
    fig = pygmt.Figure()
    if includeTopography:
        # Download grids of topography and bathymetry in the region
        grid = pygmt.datasets.load_earth_relief(resolution='01s', region=region, registration='gridline')
        # Plot the grid
        colorMap = 'geo' if useColorMap else 'gray'
        fig.grdimage(grid=grid, region=region, projection="M6i", cmap=colorMap, shading=True)
        # Add coastlines of the region to Los Angeles to a 6 inch (6i) wide map using the Mercator projection (M)
        fig.coast(shorelines='1/0.75p', region=region, projection="M6i")
    else:
        fig.coast(land='lightgray', shorelines='1/0.75p', region=region, projection="M6i")
    if includeStations:
        stations = pd.DataFrame(inputData['output_stations'], columns=['latitude', 'longitude', 'depth']).drop_duplicates(subset=['latitude', 'longitude'])
        if stationNames is None:
            stationNames = [str(i) for i in range(len(stations))]
        for i, (index, station) in enumerate(stations.iterrows()):
            fig.plot(x=station['longitude'], y=station['latitude'], style='c0.2c', fill='blue', pen='black', label='station')
            fig.text(x=station['longitude'], y=station['latitude'], text=stationNames[i], justify="BC", offset="j0.0c/0.15c", font=labelFont)
    if includeFrame:
        # Initialize the map frame with automatic tick intervals (a) and fancy frame (f, which includes geographical gridlines)
        fig.basemap(frame='af')
    fig.savefig(mapFile)
    return

if __name__ == '__main__':
    # DEBUGGING: Change the working directory to the directory of this file for debugging
    # os.chdir(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(prog='plotMap',
        description='Plot a map of the domain of the Hercules simulation using GMT.')
    parser.add_argument('--mapFile', type=str, default='map.png', help='Name of the map file')
    parser.add_argument('--herculesInputFile', '-i', type=str, default='inputfiles/parameters.in', help='Path to the Hercules input file')
    parser.add_argument('--overwriteExisting', '-o', action='store_true', help='Overwrite existing map file')
    parser.add_argument('--includeTopography', '-t', action='store_true', help='Include topography in the map')
    parser.add_argument('--useColorMap', '-c', action='store_true', help='Use a color map for the topography')
    parser.add_argument('--includeStations', '-s', action='store_true', help='Include stations in the map')
    parser.add_argument('--includeFrame', '-f', action='store_true', help='Include frame in the map')
    parser.add_argument('--stationNames', '-n', type=str, nargs='*', help='Names of the stations to \
        label in the map. Type the station names separated with spaces. If not provided, the stations will be labeled with their indices.')
    parser.add_argument('--labelFont', '-l', type=str, default='6p,black', help='Font for the station labels')
    args = parser.parse_args()

    plotMap(**vars(args))

import pygmt
import os
import argparse

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
        overwriteExisting=False, includeTopography=False, includeStations=False, 
        includeFrame=False):
    # If the map file exists, return
    if os.path.exists(mapFile) and not overwriteExisting:
        return
    inputData = getHerculesInputData(herculesInputFile)
    domainCorners = inputData['domain_surface_corners']
    region = [float(domainCorners[0][0]), float(domainCorners[2][0]), float(domainCorners[0][1]), float(domainCorners[2][1])]
    fig = pygmt.Figure()
    if includeTopography:
        # Download grids of topography and bathymetry in the region
        grid = pygmt.datasets.load_earth_relief(resolution='01s', region=region, registration='gridline')
        # Plot the grid using the geo colormap and shading
        fig.grdimage(grid=grid, region=region, projection="M6i", cmap='geo', shading=True)
        # Add coastlines of the region to Los Angeles to a 6 inch (6i) wide map using the Mercator projection (M)
        fig.coast(shorelines='1/0.5p', region=region, projection="M6i")
    else:
        fig.coast(land='lightgray', shorelines='1/0.5p', region=region, projection="M6i")
    if includeStations:
        stations = inputData['output_stations']
        for i, station in enumerate(stations):
            fig.plot(x=station[1], y=station[0], style='c0.2c', fill='blue', pen='black', label='station')
            fig.text(x=station[1], y=station[0], text=str(i), justify="BC", offset="j0.0c/0.15c", font="6p,black")
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
    parser.add_argument('--herculesInputFile', type=str, default='inputfiles/parameters.in', help='Path to the Hercules input file')
    parser.add_argument('--overwriteExisting', '-o', action='store_true', help='Overwrite existing map file')
    parser.add_argument('--includeTopography', '-t', action='store_true', help='Include topography in the map')
    parser.add_argument('--includeStations', '-s', action='store_true', help='Include stations in the map')
    parser.add_argument('--includeFrame', '-f', action='store_true', help='Include frame in the map')
    args = parser.parse_args()

    plotMap(**vars(args))
